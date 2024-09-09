# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    # 是一个多目标tracker，保存了很多个track轨迹
    # 负责调用卡尔曼滤波来预测track的新状态+进行匹配工作+初始化第一帧
    # Tracker调用update或predict的时候，其中的每个track也会各自调用自己的update或predict

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        # 调用的时候，后边的参数全部是默认的
        # metric是一个类，用于计算距离(余弦距离或马氏距离)
        self.metric = metric
        # 最大iou，iou匹配的时候使用
        self.max_iou_distance = max_iou_distance
        # 直接指定级联匹配的cascade_depth参数
        self.max_age = max_age
        # n_init代表需要n_init次数的update才会将track状态设置为confirmed
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter() # 卡尔曼滤波器
        self.tracks = []  # 保存一系列轨迹
        self._next_id = 1  # 下一个分配的轨迹id

    def predict(self):
        # 遍历每个track都进行一次预测
        for track in self.tracks:
            track.predict(self.kf)

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections):
        # 进行测量的更新和轨迹管理
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        # 1. 针对匹配上的结果
        for track_idx, detection_idx in matches:
            # track更新对应的detection
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        # 2. 针对未匹配的tracker,调用mark_missed标记
        # track失配，若待定则删除，若update时间很久也删除
        # max age是一个存活期限，默认为70帧
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 3. 针对未匹配的detection， detection失配，进行初始化
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # 得到最新的tracks列表，保存的是标记为confirmed和Tentative的track
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        # 获取所有confirmed状态的track id
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features  # 将tracks列表拼接到features列表
            # 获取每个feature对应的track id
            targets += [track.track_id for _ in track.features]
            track.features = []

        # 距离度量中的 特征集更新
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        # 主要功能是进行匹配，找到匹配的，未匹配的部分
        def gated_metric(tracks, dets, track_indices, detection_indices):
            # 功能： 用于计算track和detection之间的距离，代价函数
            #        需要使用在KM算法之前
            # 调用：
            # cost_matrix = distance_metric(tracks, detections,
            #                  track_indices, detection_indices)
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # 1. 通过最近邻计算出代价矩阵 cosine distance
            cost_matrix = self.metric.distance(features, targets)
            # 2. 计算马氏距离,得到新的状态矩阵
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # 划分不同轨迹的状态
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]



        # 进行级联匹配，得到匹配的track、不匹配的track、不匹配的detection
        '''
            !!!!!!!!!!!
            级联匹配
            !!!!!!!!!!!
            '''
        # gated_metric->cosine distance
        # 仅仅对确定态的轨迹进行级联匹配
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # 将所有状态为未确定态的轨迹和刚刚没有匹配上的轨迹组合为iou_track_candidates，
        # 进行IoU的匹配
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]   # 刚刚没有匹配上
        # 未匹配
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]   # 已经很久没有匹配上
        '''
            !!!!!!!!!!!
            IOU 匹配
            对级联匹配中还没有匹配成功的目标再进行IoU匹配
            !!!!!!!!!!!
            '''
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
