#pragma once
#ifndef SIMRANK_HPP
#define SIMRANK_HPP

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iterator>

template<typename K, typename V>
using umap = std::unordered_map<K, V>;

template<typename T>
using uset = std::unordered_set<T>;

template<typename node_t, typename float_t>
class SimRank {
public:
	SimRank(size_t K = 6, float_t C = 0.6, float_t D = 0.05);
	~SimRank();

	// Reserve space in memory for at least n nodes
	void reserve(size_t n);
	// Add an edge to the graph
	void add_edge(node_t head, node_t tail);
	// Calculate SimRank scores after adding all the edges
	void calculate_simrank(void);
	// Return the similarity score between nodes a and b
	float_t similarity(node_t a, node_t b) const;

	// Return the number of iterations
	inline size_t K(void) const { return K_; }
	// Return the decay factor
	inline float_t C(void) const { return C_; }
	// Return the delta for threshold sieving
	inline float_t D(void) const { return D_; }

	// Return the number of nodes in the graph
	inline size_t num_nodes(void) const { return node_properties_.size(); }
	// Return the number of nodes with out-degree > 0
	inline size_t num_heads(void) const { return edge_weights_.size(); }
	// Return the number of nodes with in-degree > 0
	inline size_t num_tails(void) const { return in_neighbors_.size(); }

	// Edge data accessible via edges()
	struct edge_t {
		node_t head, tail;
		float_t weight;
		inline edge_t(node_t head, node_t tail, float_t weight) : head(head), tail(tail), weight(weight) {}
	};

	class node_iterable;
	class edge_iterable;
	// Iterate over all nodes, e.g. "for (node_t x : simrank.nodes()) { ... }"
	inline const node_iterable nodes(void) const { return node_iterable(*this); }
	// Iterate over all edges, e.g. "for (SimRank::edge_t e : simrank.edges()) { ... }"
	inline const edge_iterable edges(void) const { return edge_iterable(*this); }

	// Return the out-degree of node x (the sum of the outgoing edges' weights)
	inline int out_degree(node_t x) { return node_properties_[x].out_degree; }
	// Return the in-degree of node x (the sum of the incoming edges' weights)
	inline int in_degree(node_t x) { return node_properties_[x].in_degree; }

	class out_iterable;
	class in_iterable;
	// Iterate over the out-neighbors of node x, e.g. "for (node_t y : simrank.out_neighbors(x)) { ... }"
	inline const out_iterable out_neighbors(node_t x) { return out_iterable(edge_weights_[x]); }
	// Iterate over the in-neighbors of node x, e.g. "for (node_t y : simrank.in_neighbors(x)) { ... }"
	inline const in_iterable in_neighbors(node_t x) { return in_iterable(in_neighbors_[x]); }

	// Return the weight of the edge from a to b (normalized after calling calculate_simrank())
	inline float_t edge_weight(node_t a, node_t b) { return edge_weights_[a][b]; }

private:
	struct node_prop_t {
		umap<node_t, float_t> simrank;
		float_t partial_sum;
		int in_degree, out_degree;
		inline node_prop_t(void) : simrank(), partial_sum(), in_degree(), out_degree() {}
	};

	size_t K_;
	float_t C_;
	float_t D_;

	umap<node_t, node_prop_t>           node_properties_; // {node1 -> <{node2 -> SimRank}, etc>} (node1 <= node2)
	umap<node_t, uset<node_t>>          in_neighbors_;    // {node -> {in-neighbors}}
	umap<node_t, umap<node_t, float_t>> edge_weights_;    // {head -> {tail -> weight}}

	std::vector<node_t> temp_nodes_;      // temporary node storage for calculating essential paired nodes
	std::vector<node_t> essential_nodes_; // essential paired nodes (reused in each update iteration)

	float_t *delta_; // deltas for threshold sieving

	void normalize_edges(void);
	void update_simrank_scores(node_t a, int k);

public:
	class node_iterator {
	private:
		typename umap<node_t, node_prop_t>::const_iterator pos_;
	public:
		inline node_iterator(typename umap<node_t, node_prop_t>::const_iterator pos) : pos_(pos) {}
		inline node_t operator*() { return pos_->first; }
		inline bool operator!=(const node_iterator &other) const { return pos_ != other.pos_; }
		inline const node_iterator &operator++() { ++pos_; return *this; }
	};

	class node_iterable {
	private:
		const SimRank &simrank_;
	public:
		inline node_iterable(const SimRank &s) : simrank_(s) {}
		inline const node_iterator begin(void) const { return node_iterator(simrank_.node_properties_.begin()); }
		inline const node_iterator end(void) const { return node_iterator(simrank_.node_properties_.end()); }
		inline node_iterable &operator=(const node_iterable &) { return *this; }
	};

	class edge_iterator {
		typedef typename umap<node_t, umap<node_t, float_t>>::const_iterator pos_iterator;
		typedef typename umap<node_t, float_t>::const_iterator subpos_iterator;
	private:
		pos_iterator pos_, pos_end_;
		subpos_iterator subpos_, subpos_end_;
	public:
		inline edge_iterator(pos_iterator pos, subpos_iterator subpos, pos_iterator pos_end, subpos_iterator subpos_end) :
			pos_(pos), pos_end_(pos_end), subpos_(subpos), subpos_end_(subpos_end) {}
		inline edge_t operator*() { return edge_t(pos_->first, subpos_->first, subpos_->second); }
		inline bool operator!=(const edge_iterator &other) const { return pos_ != other.pos_ || subpos_ != other.subpos_; }
		inline const edge_iterator &operator++(void) {
			++subpos_;
			if (subpos_ == pos_->second.end()) {
				++pos_;
				subpos_ = pos_ == pos_end_ ? subpos_end_ : pos_->second.begin();
			}
			return *this;
		}
	};

	class edge_iterable {
	private:
		const SimRank &simrank_;
	public:
		inline edge_iterable(const SimRank &s) : simrank_(s) {}
		inline const edge_iterator begin(void) const { return edge_iterator(simrank_.edge_weights_.begin(),
			simrank_.edge_weights_.begin()->second.begin(),
			simrank_.edge_weights_.end(),
			simrank_.edge_weights_.begin()->second.end()); }
		inline const edge_iterator end(void) const { return edge_iterator(simrank_.edge_weights_.end(),
			simrank_.edge_weights_.begin()->second.end(),
			simrank_.edge_weights_.end(),
			simrank_.edge_weights_.begin()->second.end()); }
		inline edge_iterable &operator=(const edge_iterable &) { return *this; }
	};

	class out_iterator {
	private:
		typename umap<node_t, float_t>::const_iterator pos_;
	public:
		inline out_iterator(typename umap<node_t, float_t>::const_iterator pos) : pos_(pos) {}
		inline node_t operator*() { return pos_->first; }
		inline bool operator!=(const out_iterator &other) const { return pos_ != other.pos_; }
		inline const out_iterator &operator++() { ++pos_; return *this; }
	};

	class out_iterable {
	private:
		const umap<node_t, float_t> &collection_;
	public:
		inline out_iterable(const umap<node_t, float_t> &c) : collection_(c) {}
		inline const out_iterator begin(void) const { return out_iterator(collection_.begin()); }
		inline const out_iterator end(void) const { return out_iterator(collection_.end()); }
		inline out_iterable &operator=(const out_iterable &) { return *this; }
	};

	class in_iterator {
	private:
		typename uset<node_t>::const_iterator pos_;
	public:
		inline in_iterator(typename uset<node_t>::const_iterator pos) : pos_(pos) {}
		inline node_t operator*() { return *pos_; }
		inline bool operator!=(const in_iterator &other) const { return pos_ != other.pos_; }
		inline const in_iterator &operator++() { ++pos_; return *this; }
	};

	class in_iterable {
	private:
		const uset<node_t> &collection_;
	public:
		inline in_iterable(const uset<node_t> &c) : collection_(c) {}
		inline const in_iterator begin(void) const { return in_iterator(collection_.begin()); }
		inline const in_iterator end(void) const { return in_iterator(collection_.end()); }
		inline in_iterable &operator=(const in_iterable &) { return *this; }
	};
};

template<typename node_t, typename float_t>
SimRank<node_t, float_t>::SimRank(size_t K, float_t C, float_t D) : K_(K), C_(C), D_(D),
	node_properties_(), in_neighbors_(), edge_weights_(), temp_nodes_(), essential_nodes_() {
	delta_ = new float_t[K];
}

template<typename node_t, typename float_t>
SimRank<node_t, float_t>::~SimRank() {
	delete [] delta_;
}

template<typename node_t, typename float_t>
void SimRank<node_t, float_t>::reserve(size_t n) {
	node_properties_.reserve(n);
	edge_weights_.reserve(n);
	in_neighbors_.reserve(n);
	temp_nodes_.reserve(n);
	essential_nodes_.reserve(n);
}

template<typename node_t, typename float_t>
void SimRank<node_t, float_t>::add_edge(node_t head, node_t tail) {
	node_properties_[head].out_degree++;
	node_properties_[tail].in_degree++;
	in_neighbors_[tail].insert(head);
	edge_weights_[head][tail]++;
}

template<typename node_t, typename float_t>
void SimRank<node_t, float_t>::calculate_simrank() {
	normalize_edges();
	// Calculate deltas for threshold sieving
	for (int m = 0; m < K_; m++) {
		delta_[m] = (float_t)(D_ / (K_ * pow(C_, K_ - m + 1)));
	}
	// Initialize similarity scores
	for (auto const &a_aps_p : node_properties_) {
		node_t a = a_aps_p.first;
		node_properties_[a].simrank.clear();
	}
	// Main loop: update scores for K iterations
	for (int k = 0; k < K_; k++) {
		for (auto const &a_aps_p : node_properties_) {
			node_t a = a_aps_p.first;
			int a_od = a_aps_p.second.out_degree;
			if (a_od == 0 && k < K_ - 1) { continue; }
			update_simrank_scores(a, k);
		}
	}
}

template<typename node_t, typename float_t>
float_t SimRank<node_t, float_t>::similarity(node_t a, node_t b) const {
	// similarity(a, a) == 1
	if (a == b) { return 1; }
	// similarity(a, b) == similarity(b, a), so standardize on a < b
	if (a > b) { std::swap(a, b); }
	auto a_props = node_properties_.at(a);
	auto a_b_simrank = a_props.simrank.find(b);
	if (a_b_simrank == a_props.simrank.end()) { return 0; }
	return a_b_simrank->second;
}

template<typename node_t, typename float_t>
void SimRank<node_t, float_t>::normalize_edges() {
	// Divide each edge from a to b by the in-degree of b
	for (auto &a_bws_p : edge_weights_) {
		node_t a = a_bws_p.first;
		auto &bws = a_bws_p.second;
		for (auto &b_w_p : bws) {
			node_t b = b_w_p.first;
			float_t w = b_w_p.second;
			edge_weights_[a][b] = w / node_properties_[b].in_degree;
		}
	}
}

template<typename node_t, typename float_t>
void SimRank<node_t, float_t>::update_simrank_scores(node_t a, int k) {
	// Calculate partial sums for node a's in-neighbors
	for (auto &u_ups_p : node_properties_) {
		node_t u = u_ups_p.first;
		float_t partial_sum_u = 0;
		for (node_t i : in_neighbors_[a]) {
			partial_sum_u += similarity(i, u) * edge_weights_[i][a];
		}
		node_properties_[u].partial_sum = partial_sum_u;
	}
	// Calculate essential paired nodes for node a
	essential_nodes_.clear();
	// Construct set of temporary nodes
	temp_nodes_.clear();
	for (auto const &v_vps_p : node_properties_) {
		node_t v = v_vps_p.first;
		for (node_t u : in_neighbors_[a]) {
			if (similarity(u, v) > 0) {
				temp_nodes_.push_back(v);
				break;
			}
		}
	}
	// Construct set of essential paired nodes
	for (auto const &b_bps_p : node_properties_) {
		node_t b = b_bps_p.first;
		for (node_t v : temp_nodes_) {
			if (in_neighbors_[b].find(v) != in_neighbors_[b].end()) {
				essential_nodes_.push_back(b);
				break;
			}
		}
	}
	// Main loop: account for node b's in-neighbors
	for (node_t b : essential_nodes_) {
		float_t score_a_b = 0;
		for (node_t j : in_neighbors_[b]) {
			score_a_b += node_properties_[j].partial_sum * edge_weights_[j][b];
		}
		score_a_b *= C_;
		if (score_a_b > delta_[k] || similarity(a, b) > 0) {
			node_properties_[a].simrank[b] = score_a_b;
		}
	}
}

#endif
