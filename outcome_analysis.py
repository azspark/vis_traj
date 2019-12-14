import numpy as np 
import pandas as pd 
import os 
import h5py
from geopy.distance import distance

class OutcomeAnalysis:

    def __init__(self, outcom_path, train_h5_path, test_h5_path, input_lines=None):
        # Load the datas
        if input_lines is not None:
            self.df_outcome = pd.read_csv(outcom_path, sep=' ', header=None, names=['driver_id', 'pred_id'], nrows=input_lines)
        else:
            self.df_outcome = pd.read_csv(outcom_path, sep=' ', header=None, names=['driver_id', 'pred_id'])
        self.df_outcome['traj_id'] = self.df_outcome.index
        self.df_outcome['outcome'] = self.df_outcome['driver_id'] == self.df_outcome['pred_id']
        self.train_h5 = h5py.File(train_h5_path, 'r')
        self.test_h5 = h5py.File(test_h5_path, 'r')
        # count mean train number and accuracy of each driver 
        driver_acc = self.df_outcome.groupby('driver_id')['outcome'].mean()
        df_label = self._load_train_labels()
        train_count = df_label.groupby('driver_id')['traj_id'].count()
        self.basic_info = pd.DataFrame({'train_number':train_count, 'accuracy': driver_acc})

        # group need information
        def zip_func(df):
            return list(zip(df['traj_id'], df['pred_id'], df['outcome']))
        self.driver_test_trajinfo = self.df_outcome.groupby('driver_id').apply(zip_func)
        self.driver_test_failed_trajinfo = self.df_outcome[self.df_outcome['outcome'] == False].groupby('driver_id').apply(zip_func)
        self.driver_train_trajid = df_label.groupby('driver_id')['traj_id'].apply(list)

    def get_basic_info(self):
        return self.basic_info

    def sample_driver_train_trajid(self, driver_id, num):
        total_train_num = len(self.driver_train_trajid[driver_id])
        if num > total_train_num:
            print('sample number larger than the total driver number:', total_train_num)
            num = total_train_num
        sampled_idx = np.random.choice(self.driver_train_trajid[driver_id], num, replace=False)
        trajs, middle_points = self.extract_traj(sampled_idx, 'train', get_time=False)
        return trajs, np.mean(middle_points, axis=0)

    def sample_driver_test_trajinfo(self, driver_id, num, only_failed=False):
        if only_failed:
            driver_test_info = self.driver_test_failed_trajinfo
        else:
            driver_test_info = self.driver_test_trajinfo
        total_test_num = len(driver_test_info[driver_id])
        if num > total_test_num:
            num = total_test_num
            print('sample number larger than the total test driver number')
        infos = np.array(driver_test_info[driver_id])
        sampled_info_idx = np.random.choice(len(infos), num, replace=False)
        sampled_info = infos[sampled_info_idx]
        sampled_idx = [info[0] for info in sampled_info]
        trajs, middle_points = self.extract_traj(sampled_idx, 'test', get_time=False)
        return trajs, np.mean(middle_points, axis=0), np.array(sampled_info)

    def extract_traj(self, ids, from_set='train', get_time=False):
        trajs = []
        times = []
        middle_points = []
        if from_set == 'train':
            h5 = self.train_h5
        elif from_set == 'test':
            h5 = self.test_h5
        else:
            raise Exception('Set can only be train or test')
        for tid in ids:
            one_traj = np.array(h5['trips/%d' % tid])
            one_traj_lat_lon = [(coord[1], coord[0]) for coord in one_traj]
            trajs.append(one_traj_lat_lon)
            middle_points.append(np.mean(one_traj_lat_lon, axis=0))
            if get_time:
                times.append(np.array(h5['timestamps/%d' % tid]))
        if not get_time:
            return trajs, middle_points
        else:
            return trajs, middle_points, np.array(times)

    def get_driver_detail_info(self, driver_id, from_set):
        if from_set == 'train':
            ids = self.driver_train_trajid[driver_id]
            return self.traj_dynamic_info(ids, from_set, seg_detail_info=False)
        elif from_set == 'test':
            infos = self.driver_test_trajinfo[driver_id]
            ids = [info[0] for info in infos]
            outcome = [info[2] for info in infos]
            df = self.traj_dynamic_info(ids, from_set, seg_detail_info=False)
            df['Outcome'] = outcome
            return df
        else:
            raise Exception('Set can only be train or test')

    
    def traj_dynamic_info(self, ids, from_set, seg_detail_info=False, return_origin_traj=False):
        """Get speed, time, distance information of trajecotries"""
        trajs, middle_points, times = self.extract_traj(ids, from_set, get_time=True)
        trip_distance = []  # length of each trajectory
        trip_time = []
        trip_avg_speed = []
        # Only be appended when seg_detail_infp == True
        # Feature of every trajectory segment
        traj_seg_distance = []
        traj_seg_time = []
        traj_seg_speed = []

        for index, traj in enumerate(trajs):
            assert len(traj) == len(times[index]), 'traj len should be same with timestamp'
            seg_distance = []
            for i in range(1, len(traj)):
                seg_distance.append(distance(traj[i-1], traj[i]).meters)
            seg_distance = np.array(seg_distance)
            trip_time.append(times[index][-1] - times[index][0])
            seg_times = times[index][1:] - times[index][:-1]
            # seg_times[seg_times == 0] = 1
            # if sum(seg_times <= 0) > 0:
            #     print('seg time error:', seg_times)
            # print(seg_distance, seg_distance.dtype)
            # print(seg_times, seg_times.dtype)
            seg_speed = np.divide(seg_distance, seg_times, 
                        out=np.zeros_like(seg_distance), where=seg_times!=0)
            trip_distance.append(sum(seg_distance))
            trip_avg_speed.append(np.mean(seg_speed))
            if seg_detail_info:
                traj_seg_distance.append(seg_distance)
                traj_seg_time.append(seg_times)
                traj_seg_speed.append(seg_speed)
        if not seg_detail_info:
            df = pd.DataFrame({'TripDistance': trip_distance, 
            'TripTime': trip_time, 'TripAvgSpeed': trip_avg_speed})
        else:
            df = pd.DataFrame({'TripDistance': trip_distance, 
            'TripTime': trip_time, 'TripAvgSpeed': trip_avg_speed, 'SegDistance': traj_seg_distance,
            'SegTime': traj_seg_time, 'SegSpeed': traj_seg_speed})
        df.index = ids
        if not return_origin_traj:
            return df
        else:
            return df, (trajs, middle_points, times)

    def _load_train_labels(self):
        num = self.train_h5.attrs['traj_nums']
        label = []
        for i in range(num):
            label.append(int(np.array(self.train_h5['taxi_ids/%d' % i])))
        df_label = pd.DataFrame(label, columns=['driver_id'])
        df_label['traj_id'] = df_label.index 
        # self.train_h5.close()  # close the h5 file after read
        return df_label

if __name__ == "__main__":
    pass