#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import math
import time

import networkx
from networkx.algorithms.components.connected import connected_components

import numpy as np


class CharacteristicVector():
	def __init__(self):
		self.last_time_update_of_grid = 0  # tg
		self.last_time_remove_of_grid = 0  # tm
		self.D = 0
		self.label = 'NO_CLASS'
		self.status = 'NORMAL'
		self.density_category = None
		self.category_changed_last_time = False


class DStream(object):

	def __init__(self, decay_f=0.998, dimensions_limits=None, cell=None, geo=0.1, days=1, cm=8500.0, ratecm=None, cl=0.9, beta=0.3, ztime=-1):

		self.grid_list = {}
		self.current_time = 0  # tc
		self.start_time = time.time()
		self.decay_factor = decay_f  # λ
		self.Cm = cm
		self.Cl = cl
		self.beta = beta  # β

		self.days = days
		self.ztime = ztime
		self.geo = geo
		if not dimensions_limits:
			self.dimensions_limits = [(-11.9996833333, 36.9999033333), (30.0000116667, 52.00), (1451606401, 1452075087)]  # long,lat : (-3.4992152529118434, 52.0)
		else:
			self.dimensions_limits = dimensions_limits
		self.dimensions = len(self.dimensions_limits)
		if cell:
			self.cell = cell
			self.partitions = self._get_partitions_dynamically()
		else:
			self.partitions = self._get_partitions()

		# self.partitions = [5,5,5]
		self.N = 1
		for i in range(self.dimensions):
			self.N *= self.partitions[i]
		# ###### gia to kdd-cup pou exei midenika os dimension [0.0,0.0]
		self.N = 1
		for i in range(self.dimensions):
			if self.partitions[i] != 0:
				self.N *= self.partitions[i]
		print 'partitions : ', self.partitions
		print 'N : ', self.N
		if ratecm:
			ratecm = 1 - ratecm
			self.Cm = self.N - (self.N * ratecm)
		assert self.N > self.Cm, "N must me be greater than Cm"

		self.gap = self._gap_time()
		print 'gap : ', self.gap

		# sys.exit()

		self.grid_groups = {}
		self.last_inspection = 0

		self.clusters = {}
		self.cluster_count = 0

		self.deletion = 0
		self.sporadic = 0
		self.normal = 0
		self.neighbors_dict = {}

	def _get_partitions(self):
		cellSize_geo = self.geo
		cellSize_time = self.days * 24 * 3600
		partitions = []
		self.delta = []
		for i in range(self.dimensions):
			if i < 2:
				self.delta.append(max(self.dimensions_limits[i]) - min(self.dimensions_limits[i]))
				partitions.append(int(math.ceil(self.delta[i] / cellSize_geo)))  # prosthetw tuple sto geo
			else:
				self.delta.append(max(self.dimensions_limits[i]) - min(self.dimensions_limits[i]))
				partitions.append(int(math.ceil(self.delta[i] / cellSize_time)))

		return tuple(partitions)

	def _get_partitions_dynamically(self):
		if self.ztime > -1:
			self.cell[self.ztime] = self.cell[self.ztime] * 24 * 3600
		partitions = []
		self.delta = []
		for i in range(self.dimensions):
			self.delta.append(max(self.dimensions_limits[i]) - min(self.dimensions_limits[i]))
			partitions.append(int(math.ceil(self.delta[i] / self.cell[i])))  # prosthetw tuple sto geo
		return tuple(partitions)

	def _gap_time(self):
		d0 = int(np.floor(math.log(self.Cl / self.Cm, self.decay_factor)))
		d1 = int(np.floor(math.log((self.N - self.Cm) / (self.N - self.Cl),self.decay_factor)))
		print 'dense -> sparse ', d0
		print 'sparse -> dense ', d1
		return min(d0, d1)

	def _density_threshold(self, tg):  # π(tg,t)
		top = self.Cl * (1.0 - self.decay_factor ** (self.current_time - tg + 1))
		bottom = self.N * (1.0 - self.decay_factor)
		return top / bottom

	def hash_function(self, datum):
		indicy = []

		for i in range(self.dimensions):
			# ### sto kdd-cup yparxei periptwsh ena feature na min exei timh megaluterh toy 0 kai tha exoume division by zero me to self.detla[i]=0
			if self.delta[i] == 0:
				indicy.append(0)
				continue
			lower_index = np.floor((min(self.dimensions_limits[i]) / self.delta[i]) * self.partitions[i])  # auto tha mporouse na nai public metavliti

			if lower_index >= 0:
				if i == self.ztime:

					indicy.append(int(np.floor(((float(datum[i]) - float(min(self.dimensions_limits[i]))) / (self.days * 24 * 3600)))))
				else:
					if datum[i] == self.delta[i]:
						datum[i] -= 0.000000000000001
						indicy.append(int(np.floor(float(datum[i]) / self.delta[i] * self.partitions[i]) - math.fabs(lower_index)))
						continue
					indicy.append(int(np.floor(float(datum[i]) / self.delta[i] * self.partitions[i]) - math.fabs(lower_index)))
			else:
				indicy.append(int(np.floor(float(datum[i]) / self.delta[i] * self.partitions[i]) + math.fabs(lower_index)))
		return tuple(indicy)

# ###  dokimi #######self.partitions[i]

	def reverse_hash_function(self, indicy):
		datum = []
		for i in range(self.dimensions):
			lower_index = np.floor((min(self.dimensions_limits[i]) / self.delta[i]) * self.partitions[i])
			if lower_index >= 0:
				datum.append(((indicy[i]) + math.fabs(lower_index)) / self.partitions[i] * self.delta[i])
			else:
				datum.append(((indicy[i]) - math.fabs(lower_index)) / self.partitions[i] * self.delta[i])
		return tuple(datum)
# ###  dokimi #######

	def update_charecteristic_vector(self, vector):

		vector.D = self.decay_factor ** (self.current_time - vector.last_time_update_of_grid) * vector.D + 1.0
		vector.last_time_update_of_grid = self.current_time
		return vector

	def _density_matrix(self):
		nmat = np.zeros(shape=(self.partitions[0] + 1, self.partitions[1] + 1))
		for grid, value in self.grid_list.items():

			if value.label == 'NO_CLASS':
				nmat[grid] = -1.0
			else:
				nmat[grid] = value.label
		return nmat

	def grid_density_category(self, density):
		dm = self.Cm / (self.N * (1 - self.decay_factor))
		dl = self.Cl / (self.N * (1 - self.decay_factor))

		if density > dm:
			return 'DENSE'
		elif density < dl:
			return 'SPARSE'
		elif density >= dl and density <= dm:
			return 'TRANSITIONAL'

	def detect_sporadic_grids(self):
		for grid, vector in self.grid_list.items():
			tmp_d = self.decay_factor ** (self.current_time - vector.last_time_update_of_grid) * vector.D
			if self.grid_density_category(vector.D) == 'SPARSE':
				if vector.status == 'SPORADIC':
					if vector.last_time_update_of_grid < self.last_inspection:
						vector.last_time_update_of_grid = -1  # tg
						vector.last_time_remove_of_grid = self.current_time  # tm
						vector.D = 0
						vector.label = 'NO_CLASS'
						vector.status = 'NORMAL'
						self.deletion += 1
						self.grid_list[grid] = vector
						continue
				if tmp_d < self._density_threshold(vector.last_time_update_of_grid) and self.current_time >= (1 + self.beta) * vector.last_time_remove_of_grid:  # debug:
					vector.status = 'SPORADIC'
				else:
					vector.status = 'NORMAL'

				self.grid_list[grid] = vector
		self.last_inspection = self.current_time

	def grid_neighbors(self, tmp_grid):
		neighbors = {}
		tmp = {}
		for i in range(self.dimensions):
			lst = list(tmp_grid)
			lst[i] -= 1
			tmp[i] = [tuple(lst)]
			lst[i] += 2
			tmp[i] += [tuple(lst)]

		for key, value in tmp.items():
			for v in value:
				if v in self.grid_list:
					if key not in neighbors:
						neighbors[key] = [v]
					else:
						neighbors[key] += [v]

		return neighbors

	def inside_outside_grid(self, grid):
		if len(self.neighbors_dict[grid].keys()) == self.dimensions:
			return 'INSIDE'
		else:
			return 'OUTSIDE'

	def inside_outside_grid_cluster(self, grid, clusters):
		nhgb_cl = {}
		count = 0
		if len(self.neighbors_dict[grid].keys()) == self.dimensions:

			for key in self.neighbors_dict[grid].keys():

				clusters[self.grid_list[grid].label]
				if self.neighbors_dict[grid][key] in clusters[self.grid_list[grid].label]:
					count += 1

		if count == self.dimensions:
			return 'INSIDE'
		else:
			return 'OUTSIDE'
		return
		for d, value in self.neighbors_dict[grid].items():
			for v in value:
				if v in clusters[self.grid_list[grid].label]:
					nhgb_cl[d] = v
		if len(nhgb_cl.keys()) == self.dimensions:

			return 'INSIDE'
		else:
			return 'OUTSIDE'

	def update_density_of_grids(self):
		for grid, vector in self.grid_list.items():

			vector.D = self.decay_factor ** (self.current_time - self.grid_list[grid].last_time_update_of_grid) * self.grid_list[grid].D  # + 1.0

			if self.grid_density_category(vector.D) == 'DENSE':
				if vector.density_category != 'DENSE':
					vector.category_changed_last_time = True
				else:
					vector.category_changed_last_time = False

				vector.density_category = 'DENSE'
			if self.grid_density_category(vector.D) == 'SPARSE':
				if vector.density_category != 'SPARSE':
					vector.category_changed_last_time = True
				else:
					vector.category_changed_last_time = False
				vector.density_category = 'SPARSE'
			if self.grid_density_category(vector.D) == 'TRANSITIONAL':
				if vector.density_category != 'TRANSITIONAL':
					vector.category_changed_last_time = True
				else:
					vector.category_changed_last_time = False
				vector.density_category = 'TRANSITIONAL'
			self.grid_list[grid] = vector

	def initial_clustering(self):
		cluster_count = 0
		clusters = {}
		for grid in self.grid_list:
			tmp = self.grid_neighbors(grid)

			if tmp:
				self.neighbors_dict[grid] = tmp

		self.update_density_of_grids()
		for grid in self.grid_list.keys():
			if self.grid_list[grid].density_category == 'DENSE':
				self.grid_list[grid].label = cluster_count
				clusters[cluster_count] = [grid]
				cluster_count += 1

		self.cluster_count = cluster_count
		while True:
			changes = False
			for c, grids in clusters.items():
				for grid in grids:
					if grid in self.neighbors_dict and self.outside_after_added_g(grid, clusters[self.grid_list[grid].label]):  # elegxoume an to grid einai outside me bash to cluster pou anhkei oxi olokliro to map
						for sublist in self.neighbors_dict[grid].values():
							for h in sublist:

								if self.grid_list[h].label != 'NO_CLASS':

									if self.grid_list[h].label != self.grid_list[grid].label:
										if len(clusters[self.grid_list[grid].label]) > len(clusters[self.grid_list[h].label]):
											tmp_cluster = self.grid_list[h].label
											for tmp_grid in clusters[tmp_cluster]:
												self.grid_list[tmp_grid].label = self.grid_list[grid].label
												clusters[self.grid_list[grid].label].append(tmp_grid)
											changes = True
											clusters[tmp_cluster] = []
											clusters.pop(tmp_cluster)
										else:
											tmp_cluster = self.grid_list[grid].label
											for tmp_grid in clusters[tmp_cluster]:
												self.grid_list[tmp_grid].label = self.grid_list[h].label
												clusters[self.grid_list[h].label].append(tmp_grid)
											clusters[tmp_cluster] = []
											clusters.pop(tmp_cluster)
											changes = True

								elif self.grid_density_category(self.grid_list[h].D) == 'TRANSITIONAL':
									self.grid_list[h].label = self.grid_list[grid].label
									clusters[self.grid_list[grid].label] += [h]
									changes = True

			if not changes:
				break

		for keys, values in clusters.items():
			if values == []:
				clusters.pop(keys)
		self.clusters = clusters

	def split_clusters(self, tmp_grid, cluster):
		neighbors = [tmp_grid]

		for grid in cluster:
				if tmp_grid != grid:
					tmp_ngh = []
					tmp_y = -1
					for i in range(len(self.partitions)):

						if tmp_grid[i] == grid[i]:
							tmp_ngh.append(i)
						else:
							tmp_y = i

					if len(tmp_ngh) == len(self.partitions) - 1:
						if math.fabs(tmp_grid[tmp_y] - grid[tmp_y]) == 1:
							neighbors.append(grid)
		return neighbors

	def outside_after_added_g(self, tmp_grid, cluster):
		neighbors = {}

		for grid in cluster:
			if tmp_grid != grid:
				tmp_ngh = []
				tmp_y = -1
				for i in range(len(self.partitions)):

					if tmp_grid[i] == grid[i]:
						tmp_ngh.append(i)
					else:
						tmp_y = i

				if len(tmp_ngh) == len(self.partitions) - 1:
					if math.fabs(tmp_grid[tmp_y] - grid[tmp_y]) == 1:
						if tmp_y not in neighbors:
							neighbors[tmp_y] = [grid]
						else:
							neighbors[tmp_y] += [grid]
		if len(neighbors.keys()) == self.dimensions:
			return False

		else:
			return True

	def adjust_clustering(self):

		for grid in self.grid_list.keys():

			tmp = self.grid_neighbors(grid)  # grid_neighbors

			if tmp:
				self.neighbors_dict[grid] = tmp

		self.update_density_of_grids()
		for grid in self.grid_list.keys():
			if self.grid_list[grid].density_category == 'DENSE' and self.grid_list[grid].label == 'NO_CLASS':
				self.cluster_count += 1
				self.grid_list[grid].label = self.cluster_count
				self.clusters[self.cluster_count] = [grid]
		tmp_count = 0
		self.cluster_count
		for grid, vector in self.grid_list.items():
			if vector.category_changed_last_time:

				if vector.density_category == 'SPARSE':
					if self.grid_list[grid].label != 'NO_CLASS':
						tmp_count += 1
						tmp_cluster = copy.copy(self.clusters[self.grid_list[grid].label])
						tmp_cluster.remove(grid)

						tmp_sublist_cluster = []
						for g in tmp_cluster:
							tmp_sublist_cluster.append(self.split_clusters(g, tmp_cluster))
						graph_tmp = self.to_graph(tmp_sublist_cluster)
						connected_list = list(connected_components(graph_tmp))

						self.clusters[self.grid_list[grid].label].remove(grid)

						if len(connected_list) > 1:
							self.clusters.pop(self.grid_list[grid].label)
							for split_cluster in connected_list:
								if split_cluster:
									self.cluster_count += 1
									self.clusters[self.cluster_count] = list(split_cluster)
									for split_grid in self.clusters[self.cluster_count]:
										self.grid_list[split_grid].label = self.cluster_count
						self.grid_list[grid].label = 'NO_CLASS'

				elif vector.density_category == 'DENSE':

					max_len_neighbor = -1
					max_neighbors = []

					if grid in self.neighbors_dict:

						for sublist in self.neighbors_dict[grid].values():
							for h in sublist:

								if self.grid_list[h].label != 'NO_CLASS' \
									and len(self.clusters[self.grid_list[h].label]) >= max_len_neighbor \
									and self.grid_list[h].label != self.grid_list[grid].label:
									if max_len_neighbor == len(self.clusters[self.grid_list[h].label]):
										max_neighbors.append(h)
									else:
										max_len_neighbor = len(self.clusters[self.grid_list[h].label])
										max_neighbors = [h]

						for max_neighbor in max_neighbors:
							if self.grid_list[max_neighbor].density_category == 'DENSE':

								if self.grid_list[grid].label == 'NO_CLASS':
									self.grid_list[grid].label = self.grid_list[max_neighbor].label
									self.clusters[self.grid_list[max_neighbor].label].append(grid)
									break

								elif len(self.clusters[self.grid_list[grid].label]) > max_len_neighbor:
									tmp_cluster = self.grid_list[max_neighbor].label
									for tmp_grid in self.clusters[self.grid_list[max_neighbor].label]:
										self.clusters[self.grid_list[grid].label].append(tmp_grid)

										self.grid_list[tmp_grid].label = self.grid_list[grid].label
									self.clusters.pop(tmp_cluster)
									break

								elif len(self.clusters[self.grid_list[grid].label]) <= max_len_neighbor:

									if self.grid_list[max_neighbor].label != self.grid_list[grid].label:
										tmp_label = self.grid_list[grid].label

										for tmp_grid in self.clusters[self.grid_list[grid].label]:
											self.clusters[self.grid_list[max_neighbor].label].append(tmp_grid)
											self.grid_list[tmp_grid].label = self.grid_list[max_neighbor].label

										self.clusters.pop(tmp_label)
									break

							elif self.grid_list[max_neighbor].density_category == 'TRANSITIONAL':
								updated_cluster = copy.copy(self.clusters[self.grid_list[max_neighbor].label])
								updated_cluster.append(grid)
								if self.grid_list[grid].label == 'NO_CLASS' and self.outside_after_added_g(max_neighbor, updated_cluster):

									self.grid_list[grid].label = self.grid_list[max_neighbor].label
									self.clusters[self.grid_list[max_neighbor].label].append(grid)
									break

								elif self.grid_list[grid].label != 'NO_CLASS' and len(self.clusters[self.grid_list[grid].label]) >= max_len_neighbor:  # debug: bazw = pou dn eixa

									tmp_cluster = copy.copy(self.clusters[self.grid_list[max_neighbor].label])
									tmp_cluster.remove(max_neighbor)
									tmp_sublist_cluster = []
									for g in tmp_cluster:
										tmp_sublist_cluster.append(self.split_clusters(g, tmp_cluster))
									graph_tmp = self.to_graph(tmp_sublist_cluster)
									connected_list = list(connected_components(graph_tmp))

									self.clusters[self.grid_list[max_neighbor].label].remove(max_neighbor)
									if len(connected_list) > 1:
										self.clusters.pop(self.grid_list[max_neighbor].label)
										for split_cluster in connected_list:
											if split_cluster:
												self.cluster_count += 1
												self.clusters[self.cluster_count] = list(split_cluster)
												for split_grid in self.clusters[self.cluster_count]:
													self.grid_list[split_grid].label = self.cluster_count
									self.grid_list[max_neighbor].label = self.grid_list[grid].label
									self.clusters[self.grid_list[grid].label].append(max_neighbor)

									break

				elif vector.density_category == 'TRANSITIONAL':

					neighbor_clusters = {}
					if grid in self.neighbors_dict:
						for neighbor in self.neighbors_dict[grid].values():
							for sublist in neighbor:
								if self.grid_list[sublist].label != 'NO_CLASS' and self.grid_list[sublist].label != self.grid_list[grid].label:  # debug elegxw na min einai sto idio cluster
									if self.grid_list[sublist].label not in neighbor_clusters:
										neighbor_clusters[self.grid_list[sublist].label] = self.clusters[self.grid_list[sublist].label]

					if neighbor_clusters:
						while neighbor_clusters:
							max_neighbor = self.getmaxflow(neighbor_clusters)  # debug: gia na briskw to cluster me to megalitero lenght stoixiwn

							updated_cluster = copy.copy(self.clusters[max_neighbor])
							if grid not in updated_cluster:  # elegxw an uparxei idi to grid giati auto dimiourgei ta dipla
								updated_cluster.append(grid)
								if self.outside_after_added_g(grid, updated_cluster):

									if self.grid_list[grid].label != 'NO_CLASS':

										tmp_cluster = copy.copy(self.clusters[self.grid_list[grid].label])
										tmp_cluster.remove(grid)

										tmp_sublist_cluster = []
										for g in tmp_cluster:
											tmp_sublist_cluster.append(self.split_clusters(g, tmp_cluster))
										graph_tmp = self.to_graph(tmp_sublist_cluster)
										connected_list = list(connected_components(graph_tmp))
										self.clusters[self.grid_list[grid].label].remove(grid)
										if len(connected_list) > 1:
											self.clusters.pop(self.grid_list[grid].label)
											for split_cluster in connected_list:
												if split_cluster:
													self.cluster_count += 1
													self.clusters[self.cluster_count] = list(split_cluster)
													for split_grid in self.clusters[self.cluster_count]:
														self.grid_list[split_grid].label = self.cluster_count

									self.grid_list[grid].label = max_neighbor
									self.clusters[max_neighbor].append(grid)
									break
								else:
									neighbor_clusters.pop(max_neighbor)
							else:
								break

		for keys, values in self.clusters.items():
			if values == []:
				self.clusters.pop(keys)


# ######## graph ################################

	def to_graph(self, l):
		g = networkx.Graph()
		for part in l:
			g.add_nodes_from(part)
			g.add_edges_from(self.to_edges(part))
		return g

	def to_edges(self, l):
		it = iter(l)
		last = next(it)

		for current in it:
			yield last, current
			last = current

# ######## graph ################################

# ####### Get longest element in Dict ##########
	def getmaxflow(self, flows):
		maks = max(flows, key=lambda k: len(flows[k]))
		return maks  # ( len , key)

# ####### Get longest element in Dict ##########
