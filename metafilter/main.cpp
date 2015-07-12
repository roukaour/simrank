#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

#include "../simrank.hpp"

typedef SimRank<int, float> MeFiSimRank;

void read_favorites(MeFiSimRank &mfsr, const char *filename) {
	std::ifstream ifs(filename);
	std::string line;
	std::getline(ifs, line); // timestamp
	std::getline(ifs, line); // headers
	while (std::getline(ifs, line)) {
		// fav_id \t faver \t favee \t ...
		std::istringstream iss(line);
		int fav_id, faver, favee;
		iss >> fav_id >> faver >> favee;
		mfsr.add_edge(faver, favee);
	}
}

void print_statistics(MeFiSimRank &mfsr) {
	// Print SimRank parameters
	std::cout << "SimRank: K = " << mfsr.K() << ", C = " << mfsr.C() << ", D = " << mfsr.D() << std::endl;

	// Print user-related totals
	std::cout << mfsr.num_nodes() << " total users" << std::endl;
	std::cout << mfsr.num_heads() << " users are favers" << std::endl;
	std::cout << mfsr.num_tails() << " users are favees" << std::endl;

	// Calculate favorite-related totals
	size_t num_edges = 0;
	size_t total_weight = 0;
	size_t max_weight = 0; int max_weight_faver = 0, max_weight_favee = 0;
	std::map<size_t, size_t> weight_histogram;
	for (auto edge : mfsr.edges()) {
		num_edges++;
		size_t weight = (size_t)edge.weight;
		total_weight += weight;
		if (weight > max_weight) {
			max_weight = weight;
			max_weight_faver = edge.head;
			max_weight_favee = edge.tail;
		}
		weight_histogram[weight]++;
	}
	// Print favorite-related totals
	std::cout << total_weight << " total favorites" << std::endl;
	std::cout << num_edges << " unique faver-favee pairs" << std::endl;

	// Print statistical averages
	std::cout << "average " << (total_weight / mfsr.num_heads()) << " favorites given per user" << std::endl;
	std::cout << "average " << (total_weight / mfsr.num_tails()) << " favorites received per user" << std::endl;
	std::cout << "average " << (total_weight / num_edges) << " favorites per faver-favee pair" << std::endl;

	// Calculate statistical maxima
	size_t max_faves_given = 0; int max_faves_given_user = 0;
	size_t max_faves_received = 0; int max_faves_received_user = 0;
	size_t max_favee_count = 0; int max_favee_count_user = 0;
	size_t max_faver_count = 0; int max_faver_count_user = 0;
	for (int user : mfsr.nodes()) {
		size_t num_faves_given = mfsr.out_degree(user);
		if (num_faves_given > max_faves_given) {
			max_faves_given = num_faves_given;
			max_faves_given_user = user;
		}

		size_t num_faves_received = mfsr.in_degree(user);
		if (num_faves_received > max_faves_received) {
			max_faves_received = num_faves_received;
			max_faves_received_user = user;
		}

#pragma warning(disable: 4189) // '_': local variable is initialized but not referenced
		size_t num_favee_count = 0;
		for (int _ : mfsr.out_neighbors(user)) { num_favee_count++; }
		if (num_favee_count > max_favee_count) {
			max_favee_count = num_favee_count;
			max_favee_count_user = user;
		}

		size_t num_faver_count = 0;
		for (int _ : mfsr.in_neighbors(user)) { num_faver_count++; }
		if (num_faver_count > max_faver_count) {
			max_faver_count = num_faver_count;
			max_faver_count_user = user;
		}
	}
	// Print maxima
	std::cout << "maximum " << max_faves_given << " favorites given by user #" << max_faves_given_user << std::endl;
	std::cout << "maximum " << max_faves_received << " favorites received by user #" << max_faves_received_user << std::endl;
	std::cout << "maximum " << max_weight << " favorites given by user #" << max_weight_faver << " to user #" << max_weight_favee << std::endl;
	std::cout << "maximum " << max_favee_count << " favees for user #" << max_favee_count_user << std::endl;
	std::cout << "maximum " << max_faver_count << " favers of user #" << max_faver_count_user << std::endl;
}

int main(int argc, char *argv[]) {
	MeFiSimRank mefisimrank(6, 0.6f);
	const char *filename = argc > 1 ? argv[1] : "favoritesdata.txt";
	read_favorites(mefisimrank, filename);
	print_statistics(mefisimrank);
	// TODO: run SimRank on the significant subset of data
	return 0;
}
