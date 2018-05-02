#!/usr/bin/env python
# -*- coding: utf-8 -*-
activate_this = "./env/bin/activate_this.py"
execfile(activate_this, dict(__file__=activate_this))
import csv

import time

import numpy as np

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

from d_stream import CharacteristicVector
from d_stream import DStream

'''
http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html dataset tou paper.
data consisting of 7 weeks of network-based intrusions inserted in the normal data and 2 weeks of network-based intrusions and normal data for a total of 4,999,000 connec- tion records described by 42 characteristics. KDD CUP99 has been used in [14, 17, 24, 27] 
and it is converted into data stream by taking the data input order as the order of streaming.
pdf : A Fast Density-Based Clustering Algorithm for Real-Time Internet of Things Stream
'''


def noise_generator(noise, x_min, x_max, y_min, y_max):
	label_zero = np.empty(	noise)
	label_zero.fill(0)
	np.random.seed(40)
	no_structure, label_n = np.random.uniform(x_min, x_max, [noise, 1]), None
	no_structure1, label_n = np.random.uniform(y_min, y_max, [noise, 1]), None
	no_structure = np.c_[no_structure, no_structure1, label_zero]
	return no_structure


def custom_dataset():

	n_samples = 100000 / 5
	noise = int(n_samples * 0.1)
	noise_counter = 0
	params = {
		'noise': noise,
		'x_min': -2.5,
		'x_max': 4.7,
		'y_min': -11.2,
		'y_max': 3.5
	}

	# ======== moons ========

	noisy_moons_data, label_m = datasets.make_moons(n_samples=n_samples * 2, shuffle=True, noise=0.11)
	moon_a = []
	moon_b = []
	countx = 0

	for n in noisy_moons_data:

		if label_m[countx] == 0:
			moon_a.append([n[0], n[1], 1.0])

		else:
			moon_b.append([n[0], n[1], 2.0])
			moon_b[-1][1] += -2.2

		countx += 1

	# ======== moon a ========
	noise_points = noise_generator(**params)
	noise_counter += len(noise_points)
	noisy_moon_a = np.concatenate((moon_a, noise_points), axis=0)
	np.random.shuffle(noisy_moon_a)

	# ======== moon b ========
	noise_points = noise_generator(**params)
	noise_counter += len(noise_points)
	noisy_moon_b = np.concatenate((moon_b, noise_points), axis=0)
	np.random.shuffle(noisy_moon_b)
	dataset = np.concatenate((noisy_moon_a, noisy_moon_b), axis=0)

	# ======== end moons ========

	# ======== blob 	=========

	centers = [[3.3, .9]]
	blob, label = datasets.make_blobs(n_samples=n_samples, centers=centers, shuffle=True, cluster_std=0.32, random_state=0)
	label_blob = np.empty(n_samples)
	label_blob.fill(3)
	blob = np.c_[blob, label_blob]

	noise_points = noise_generator(**params)
	noise_counter += len(noise_points)

	noisy_blob = np.concatenate((blob, noise_points), axis=0)

	np.random.shuffle(noisy_blob)

	dataset = np.concatenate((dataset, noisy_blob), axis=0)

	# ======== end blob =========

	# ======== circles ========

	noisy_circles_data, label_c = datasets.make_circles(
		n_samples=n_samples * 2,
		shuffle=True, factor=.99,
		noise=.1
	)
	noisy_circles_data1, label_c1 = datasets.make_circles(
		n_samples=n_samples,
		shuffle=True, factor=.99,
		noise=.1
	)
	# ======== position of circles ========
	circle_a = []
	circle_b = []
	countx = 0
	x_ax = 2
	y_ax = -8
	for n in noisy_circles_data:
		if label_c[countx] == 0:
			circle_a.append([n[0], n[1], 4.0])
			circle_a[-1][0] += x_ax  # x
			circle_a[-1][1] += y_ax  # y
		else:
			circle_b.append([n[0], n[1], 4.0])
			circle_b[-1][0] += -1.9 + x_ax  # x
			circle_b[-1][1] += y_ax  # y
		countx += 1

	circles = np.concatenate((circle_a, circle_b), axis=0)
	noise_points = noise_generator(**params)
	noise_counter += len(noise_points)

	noisy_circles = np.concatenate((circles, noise_points), axis=0)
	np.random.shuffle(noisy_circles)

	dataset = np.concatenate((dataset, noisy_circles), axis=0)
	print 'noise : ', noise_counter, '\n'

	# ======== end circles ========
	# np.random.shuffle(dataset)

	# ======== normalize data ========

	scaler = MinMaxScaler((0, 1))
	scaled_dataset = scaler.fit_transform(dataset)

	# ======== end normalize data ========

	# ======== dimensions ========

	dimensions = [[9999.0, -999.0] for i in range(2)]

	for feature in scaled_dataset:
		for index, value in enumerate(feature[:2]):

			if dimensions[index][0] > value:
				dimensions[index][0] = value
			if dimensions[index][1] < value:
				dimensions[index][1] = value
	# ======== end dimensions ========
	return scaled_dataset, dimensions


def hash_label(label):
	dict_table = {
		'normal.': 0,
		'back.': 1,
		'land.': 1,
		'neptune.': 1,
		'pod.': 1,
		'smurf.': 1,
		'teardrop.': 1,
		'buffer_overflow.': 2,
		'loadmodule.': 2,
		'perl.': 2,
		'rootkit.': 2,
		'ftp_write.': 3,
		'guess_passwd.': 3,
		'imap.': 3,
		'multihop.': 3,
		'phf.': 3,
		'spy.': 3,
		'warezclient.': 3,
		'warezmaster.': 3,
		'ipsweep.': 4,
		'nmap.': 4,
		'portsweep.': 4,
		'satan.': 4
	}
	return dict_table[label]


def kdd_dataset(features_select=None):
	flag_break = 0
	features = []
	with open('./kddcup.data_10_percent_corrected', 'rb') as csvfile:
		col = csv.reader(csvfile, delimiter=',', quotechar='|')
		for rows in col:
			if flag_break == 1000:
				break
			flag_break += 1

			feat_list = []
			for index, feature in enumerate(rows):
					if index not in [1, 2, 3, 6, 11, 20, 21]:
						if index != 41:
							feat_list.append(float(feature))
						else:
							feat_list.append(hash_label(feature))
			features.append(feat_list)
	print len(features_select)
	dimensions = [[9999.0, -999.0] for i in range(len(features_select))]

	scaler = MinMaxScaler((0, 1))
	scaled_x = scaler.fit_transform(features)

	for feature in scaled_x:
		for index, value in enumerate(feature[:len(features_select)]):
			if dimensions[index][0] > value:
				dimensions[index][0] = value
			if dimensions[index][1] < value:
				dimensions[index][1] = value

	return scaled_x, dimensions


'''
[0]	duration: continuous. [1]	protocol_type: symbolic. [2]	service: symbolic. [3]	flag: symbolic. [4]	src_bytes: continuous.
[5]	dst_bytes: continuous. [6]	land: symbolic. [7]	wrong_fragment: continuous. [8]	urgent: continuous. [9]	hot: continuous.
[10]num_failed_logins: continuous. [11]logged_in: symbolic. [12]num_compromised: continuous. [13]root_shell: continuous.
[14]su_attempted: continuous. [15]num_root: continuous. [16]num_file_creations: continuous. [17]num_shells: continuous.
[18]num_access_files: continuous. [19]num_outbound_cmds: continuous. [20]is_host_login: symbolic. [21]is_guest_login: symbolic.
[22]count: continuous. [23]srv_count: continuous. [24]serror_rate: continuous. [25]srv_serror_rate: continuous.
[26]rerror_rate: continuous. [27]srv_rerror_rate: continuous. [28]same_srv_rate: continuous. [29]diff_srv_rate: continuous.
[30]srv_diff_host_rate: continuous. [31]dst_host_count: continuous. [32]dst_host_srv_count: continuous.
[33]dst_host_same_srv_rate: continuous.[34]dst_host_diff_srv_rate: continuous. [35]dst_host_same_src_port_rate: continuous.
[36]dst_host_srv_diff_host_rate: continuous.
[37]dst_host_serror_rate: continuous. [38]dst_host_srv_serror_rate: continuous. [39]dst_host_rerror_rate: continuous.
[40]dst_host_srv_rerror_rate: continuous.
symbolic   : 1,2,3,6,11,20,21
continuous : [0, 4, 5, 7, 8, 9, 10,  12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
'''
# ola ta continuous features
features = [
	0, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16,
	17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29,
	30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40
]
# ola ta continuous features

# proteinomena features gia to dataset na anagnorizei ta clusters
features = [22, 23, 24, 25, 28, 29, 31, 32, 33, 34, 35, 37, 38, 41]
features = [22, 28, 31, 32, 33]
# proteinomena features gia to dataset na anagnorizei ta clusters

# trajectories, dimensions = kdd_dataset(features)
trajectories, dimensions = custom_dataset()
dataset_info = 'custom_dataset'
len_dataset = len(trajectories)
print 'dimensions', dimensions
# dataset_info =  'kdd_dataset'
cells = [[0.025, 0.025], [0.04, 0.04], [0.05, 0.05]]

for defay_factor in [.45, .5, .55, .6, .65, .7]:
	export_ssq_list = []
	export_purity_list = []
	export_clusters_list = []
	for cell in cells:
		dstream_params = {
			'decay_f': defay_factor,
			'dimensions_limits': dimensions,
			'cell': cell,
			'cm': 3.0,
			'cl': 0.8,
			'beta': 0.3,
			# 'geo':2,
			# 'days' : 0.25,
			'ratecm': 0.89
		}
		alg = DStream(**dstream_params)
		print alg.delta, 'delta'
		print 'Dm ', alg.Cm / (alg.N * (1 - alg.decay_factor)), 'Cm ', alg.Cm, 'alg.N ', alg.N, 'alg.decay_factor ', alg.decay_factor
		print 'Dl is %.10f' % (alg.Cl / (alg.N * (1 - alg.decay_factor))), 'Cl ', alg.Cl

		changed = False
		flag = 0
		flag_init = False

		# ##### custome dataset algorithm ######
		timez = time.time()

		stream_speed = 1000
		horizon = 5
		raw_data = []
		for row in trajectories:

			raw_data.append(row)
			g = alg.hash_function(row)
			if g not in alg.grid_list:
				alg.grid_list[g] = CharacteristicVector()
			alg.grid_list[g] = alg.update_charecteristic_vector(alg.grid_list[g])

			if not flag_init and alg.current_time == alg.gap:
				flag_init = True

				print 'init', alg.current_time
				alg.initial_clustering()
				print '############# Clusters ###########'
				print alg.clusters

			elif alg.current_time % alg.gap == 0 and changed and flag_init:
				alg.detect_sporadic_grids()
				alg.adjust_clustering()
				print alg.current_time, 'adjust'
				print '############# Clusters ###########'
				print alg.clusters

			if len(raw_data) > horizon * stream_speed:
				del raw_data[:stream_speed]
			flag += 1
			changed = False

			if flag % stream_speed == 0:
				alg.current_time += 1
				changed = True

		print 'current : ', alg.current_time, 'duration : ', time.time() - timez

		print '############# Clusters ###########'
