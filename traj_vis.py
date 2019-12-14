import folium
from outcome_analysis import OutcomeAnalysis
import numpy as np

class TrajVisualizer:
    def __init__(self, outcome_path, train_h5_path, test_h5_path, input_lines=None):
        self.oa = OutcomeAnalysis(outcome_path, train_h5_path, test_h5_path, input_lines)

    def show_driver_training_traj(self, driver_id, number=20, zoom=12, with_marker=True):
        """Show the driver trajectory from train data"""
        trajs, middle_point = self.oa.sample_driver_train_trajid(driver_id, number)
        m = self.basic_map(zoom, middle_point)
        for traj in trajs:
            folium.PolyLine(traj, color='blue').add_to(m)
            if with_marker:
                folium.Marker(location=traj[0]).add_to(m)
        return m

    def show_tested_traj(self, driver_id, number=20, zoom=12, with_marker=True, only_failed=False):
        """Show the driver's tested trajectory with different color 
        predicted true with blue color and wrong with red color"""
        trajs, middle_point, info = self.oa.sample_driver_test_trajinfo(driver_id, number, only_failed=only_failed)
        # info: [traj_id, pred_id, outcome]
        m = self.basic_map(zoom, middle_point)
        for idx, traj in enumerate(trajs):
            color = 'red' if info[idx, 2] == False else 'blue'
            popup_info = '<i>Test traj%d: predicted as driver%d</i>' % (info[idx, 0], info[idx, 1])
            folium.PolyLine(traj, color=color, popup=popup_info).add_to(m)
            if with_marker:
                folium.Marker(location=traj[0]).add_to(m)
        return m

    def show_traj_details(self, traj_ids, from_set, zoom=13):
        """Show trajectory details given trajectory id"""
        info_df, (trajs, middle_points, times) = self.oa.traj_dynamic_info(traj_ids, from_set, 
                    seg_detail_info=True, return_origin_traj=True)
        print(info_df.columns)
        m = self.basic_map(zoom, np.mean(middle_points, axis=0))
        for idx, traj in enumerate(trajs):
            folium.PolyLine(traj, color='blue').add_to(m)
            seg_speeds = info_df.loc[traj_ids[idx], 'SegSpeed']
            seg_Distance = info_df.loc[traj_ids[idx], 'SegDistance']
            for i in range(1, len(traj)):
                tooltip = 'Speed:%f, Distance:%f' % (seg_speeds[i-1], seg_Distance[i-1])
                folium.Marker(location=traj[i], tooltip=tooltip).add_to(m)
        return m

    def get_driver_detail_info(self, driver_id, from_set):
        """
        Get information of driver's train or test trajectories in form of pandas dataframe

        Return Dataframe consists of:
        | traj_id | driving distance | driving time | avs_speed | start date(year:month:day:hour:minute) |
        if from_set == test, classification outcome will also be given
        TODO: add appear Time information, use pandas to speed up
        """
        return self.oa.get_driver_detail_info(driver_id, from_set)

    def basic_map(self, zoom, center):
        return folium.Map(location=center, zoom_start=zoom)

    def get_basic_info(self):
        return self.oa.get_basic_info()