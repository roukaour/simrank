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
	size_t num_pairs = 0;
	size_t total_favorites = 0;
	size_t max_favorites = 0; int max_faves_faver = 0, max_faves_favee = 0;
	std::map<size_t, size_t> favorites_histogram;
	for (auto edge : mfsr.edges()) {
		num_pairs++;
		size_t pair_favorites = (size_t)edge.weight;
		total_favorites += pair_favorites;
		if (pair_favorites > max_favorites) {
			max_favorites = pair_favorites;
			max_faves_faver = edge.head;
			max_faves_favee = edge.tail;
		}
		favorites_histogram[pair_favorites]++;
	}
	// Print favorite-related totals
	std::cout << total_favorites << " total favorites" << std::endl;
	std::cout << num_pairs << " unique faver-favee pairs" << std::endl;

	// Print statistical averages
	std::cout << "average " << (total_favorites / mfsr.num_heads()) << " favorites given per user" << std::endl;
	std::cout << "average " << (total_favorites / mfsr.num_tails()) << " favorites received per user" << std::endl;
	std::cout << "average " << (total_favorites / num_pairs) << " favorites per faver-favee pair" << std::endl;

	// Calculate statistical maxima
	size_t max_faves_given = 0; int max_faves_given_user = 0;
	size_t max_faves_received = 0; int max_faves_received_user = 0;
	size_t max_favee_count = 0; int max_favee_count_user = 0;
	size_t max_faver_count = 0; int max_faver_count_user = 0;
	std::map<size_t, size_t> favers_histogram, favees_histogram;
	for (int user : mfsr.nodes()) {
		size_t num_faves_given = (size_t)mfsr.out_degree(user);
		if (num_faves_given > max_faves_given) {
			max_faves_given = num_faves_given;
			max_faves_given_user = user;
		}
		favers_histogram[num_faves_given]++;

		size_t num_faves_received = (size_t)mfsr.in_degree(user);
		if (num_faves_received > max_faves_received) {
			max_faves_received = num_faves_received;
			max_faves_received_user = user;
		}
		favees_histogram[num_faves_received]++;

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
	std::cout << "maximum " << max_favorites << " favorites given by user #" << max_faves_faver << " to user #" << max_faves_favee << std::endl;
	std::cout << "maximum " << max_favee_count << " favees for user #" << max_favee_count_user << std::endl;
	std::cout << "maximum " << max_faver_count << " favers of user #" << max_faver_count_user << std::endl;

	// Print histogram data
	// num favorites \t count givers (favers) \t count receivers (favees) \t count faver-favee pairs (connections)
	std::cout << std::endl;
	std::cout << "num_faves\tfaver_count\tfavee_count\tpair_count" << std::endl;
	size_t max_count = std::max(std::max(favers_histogram.rbegin()->first, favees_histogram.rbegin()->first), favorites_histogram.rbegin()->first);
	for (size_t n = 0; n <= max_count; n++) {
		std::cout << n << "\t" << favers_histogram[n] << "\t" << favees_histogram[n] << "\t" << favorites_histogram[n] << std::endl;
	}
}

int main(int argc, char *argv[]) {
	MeFiSimRank mefisimrank(6, 0.6f);
	const char *filename = argc > 1 ? argv[1] : "favoritesdata.txt";
	read_favorites(mefisimrank, filename);
	print_statistics(mefisimrank);
	// TODO: run SimRank on the significant subset of data
	return 0;
}
