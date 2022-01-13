/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games (TCG 2021)
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include <fstream>
#include <memory>
#include <chrono>
#include <assert.h>
#include <cstdio>
#include <cmath>
#include <torch/torch.h>
#include <torch/script.h>
bool DEBUG = false;
bool SELECTION_DEBUG = true;
bool EXPAND_DEBUG = true;
bool EVALUATION_DEBUG = true;
bool UPDATE_DEBUG = true;
bool TRAINING = false;
class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	// virtual action take_action(const board& b) { return action(); }
	virtual std::tuple<action, std::map<size_t, size_t>  > take_action(const board& b) { return std::make_tuple(action(), policy_labels); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
	std::map<size_t, size_t> policy_labels;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}

	virtual std::tuple<action, std::map<size_t, size_t> > take_action(const board& state) {
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return std::make_tuple(move, policy_labels);
		}
		return std::make_tuple(action(), policy_labels);
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
};

board::piece_type next(board::piece_type color) { 
	if (color == board::black) { return board::white; }
	else if (color == board::white) { return board::black; }
	return board::unknown;	
}
static time_t millisec() {
	auto now = std::chrono::system_clock::now().time_since_epoch();
	return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}


class AlphaZeroPlayer : public player {
public:
	AlphaZeroPlayer(const std::string& args = "") : player("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);

		if(args.find("mcts") != std::string::npos) {
			use_mcts = true;
			if(args.find("N") != std::string::npos) {
				if (std::stoi(sim_count_limit_property()) > 0) { sim_count_limit = std::stoi(sim_count_limit_property()); }
			}
			if(args.find("T") != std::string::npos) {
				if (std::stoi(time_limit_property()) > 0) { time_limit = std::stof(time_limit_property()); }
			}
		}
		
		// if (DEBUG) { std::cout << "sim_count_limit:" << sim_count_limit << std::endl; }
		std::cout << "sim_count_limit:" << sim_count_limit << std::endl;
		std::cout << "time_limit:" << time_limit << std::endl;
	}
	virtual std::string sim_count_limit_property() const { return property("N"); }
	virtual std::string time_limit_property() const { return property("T"); }
	virtual std::string search() const { return property("search"); }
public:
	void load_model(const std::string& model_path) {
		try {
			// std::cout << "torch::cuda::is_available():" << torch::cuda::is_available() << std::endl;
			net = torch::jit::load(model_path);
			net.to(torch::kCUDA);
			net.eval();
		}
		catch (const c10::Error &e) {
			std::cerr << "error loading " << model_path << std::endl;
			std::cerr << e.what() << std::endl;
			assert(false);
		}
		std::cout << "OK\n";
	}
	virtual std::tuple<action, std::map<size_t, size_t> > take_action(const board& state) {
		board::piece_type color = state.info().who_take_turns;
		if (not state.has_legal_move(color)) { return std::make_tuple(action(), policy_labels); }
		Node root;
		root.parent = nullptr;
		root.evaluation(state, net);
		if (TRAINING) { root.add_dirichlet_noise(81, engine); }
		if (DEBUG) { printf("\t\troot:%p\n", &root); }
		time_t tic = millisec();
		time_t toc = millisec();
		size_t sim_count = 0;
		for (size_t i = 0; (i < sim_count_limit) and (toc - tic < time_limit-5); i ++) {
			sim_count += 1;
			Node* node = &root;
			board s(state);
			board::piece_type color = s.info().who_take_turns;
			node->color = next(color);
			while (node->num_children > 0) {
				node = node->select();
				assert(node != nullptr);
				board::reward r = s.place(node->move);
				if (r != board::legal) { 
					auto legal_moves = s.get_legal_move(color);
					return std::make_tuple(action::place(legal_moves[0], who), policy_labels);
				}
				if (DEBUG) {
					if (r != board::legal) {
						std::cout << s << std::endl;
						std::cout << "reward:" << r << "\tnode->move.i:" << node->move.i << "\tnode->move:" << node->move << std::endl; 
					}
				}
				assert(r == board::legal);
			}
			bool expand_success;
			node->expand(s, expand_success);
			// std::uniform_int_distribution<int> uniform(0, node->num_children-1);
			// int rand_idx = uniform(engine);
			// node = &(node->children[rand_idx]);
			// std::cout << s << std::endl;
			node->evaluation(s, net);
			float value = node->V;
			while (node != nullptr) {
				node = node->update(value);
				value = -value;
			}
			toc = millisec();
		}
		auto visit_distr = root.get_visit_distr();
		if (DEBUG) { 
			std::cout << "visit distribution ";
			for (auto& it : visit_distr) {
				std::cout << root.children[it.first].move << ":" << it.second << " ";
			}
			std::cout << std::endl;
		}
		// std::cout << "visit distribution ";
		// for (auto& it : visit_distr) {
		// 	std::cout << root.children[it.first].move << ":" << it.second << " ";
		// }
		// std::cout << std::endl;
		size_t max_idx = std::max_element(visit_distr.begin(), visit_distr.end(),
			[](const std::pair<size_t, size_t>& p1, const std::pair<size_t, size_t>& p2) {
				return p1.second < p2.second; })->first;
		board::point best_move = root.children[max_idx].move;
		std::map<size_t, size_t> visit_distr_;
		if (TRAINING) {
			for (const auto &it : visit_distr) {
				// int x = root.children[it.first].move.x;
				// int y = root.children[it.first].move.y;
				int i = root.children[it.first].move.i;
				visit_distr_[i] = it.second;
			}
		}
		
		// std::cout << "visit distribution_ ";
		// for (auto& it : visit_distr_) {
		// 	std::cout << board::point(it.first) << ":" << it.second << " ";
		// }
		// std::cout << std::endl;
		return std::make_tuple(action::place(best_move, who), visit_distr_);
	}
	
	virtual bool check_for_win(const board& b) {
		if (not b.has_legal_move(b.info().who_take_turns)) { return true; }
		else { return false; } 
	}

private:
	bool use_mcts = true;
	size_t sim_count_limit = 10000;
	float time_limit = 10000;	// in millisec
	std::vector<action::place> space;
	board::piece_type who;
	size_t turn = 0;
	torch::jit::script::Module net;
	// torch::nn::Sequential net;

private:
	class Node {
	friend class AlphaZeroPlayer;
	private:
		size_t num_children = 0;
		// Node* children;
		std::unique_ptr<Node[]> children;
		board::point move = board::illegal_pass;
		Node* parent;
		bool is_leaf = false;
		board::piece_type color = board::empty;
		float win_value = 0;
		uint visit_count = 0;
		float puct_value;
		float Q = 0;
		float V = 0;
		std::array<float, 81> pi;
		
	public:
		Node* get_parent() { return parent; }
	public:
		Node* select() {
			if (children == nullptr) { return nullptr; }
			if (DEBUG) { std::cout << "\tselect()" << std::endl; }
			// float max_uct_value = std::numeric_limits<float>::min();
			float max_uct_value = -100000000;
			size_t max_index = -1;
			float C = 2.0;
			if (DEBUG) { std::cout << "\t\tnum_children:" << num_children << std::endl; }
			for (size_t i = 0; i < num_children; i++) {
				auto& child = children[i];
				// PUCT
				// float U = C * std::sqrt(std::log(visit_count + 1) / (child.visit_count + 1));
				float U = C * pi[child.move.i] * std::sqrt(visit_count) / (1 + child.visit_count);

				child.puct_value = Q + U;
				if (DEBUG) { 
					if (Q != 0) {
						std::cout << "\tQ:" << Q << "\tU:" << U << std::endl;
						std::cout << "\tpi:" << pi[child.move.i] << "\tvisit_count:" << visit_count << "\tchild.visit_count:" << child.visit_count << std::endl;
					}
				}
				if (max_uct_value < child.puct_value) {
					max_uct_value = child.puct_value;
					max_index = i;
				}
			}
			if (DEBUG) { std::cout << "max_index:" << max_index << std::endl; }
			if (DEBUG) { printf("\t\tnode %p -> node %p\n", this, &(children[max_index])); }
			return &(children[max_index]);
		}
		void expand(board state, bool& is_success) {
			if (is_leaf or visit_count == 0) {
				is_success = false;
				return;
			}
			auto legal_moves = state.get_legal_move(next(color));
			// std::cout << "legal move:";
			// for (const auto &move : legal_moves) { std::cout << move << " "; }
			// std::cout << std::endl;
			num_children = legal_moves.size();
			// std::cout << "\texpand num_children:" << num_children << std::endl;
			if (num_children == 0) {
				is_leaf = true;
				is_success = false;
				return;
			}
			if (DEBUG) { std::cout << "\texpand()" << std::endl; }
			children = std::make_unique<Node[]>(num_children);
			for (size_t i = 0; i < num_children; i++) {
				children[i].color = next(color);
				children[i].move = legal_moves[i];
				children[i].parent = this;
			}
			is_success = true;
		}
		void evaluation(const board &b, torch::jit::script::Module &net) {
			torch::Tensor inputs = torch::from_blob(b.observation_tensor().data(), {1, 10, 9, 9}).to(torch::kCUDA);
			const auto &outputs = net.forward({inputs});
			// std::cout << "outputs:" << outputs << std::endl;
			const auto &p_tensor = torch::softmax(outputs.toTuple()->elements()[0].toTensor(), 1).to(torch::kCPU); 	// (1, 81)
			// std::cout << "p_tensor.size(0):" << p_tensor.size(0) << std::endl;
			// std::cout << "p_tensor.size(1):" << p_tensor.size(1) << std::endl;
			const auto p_accessor = p_tensor.accessor<float, 2>();
			const auto &v_tensor = outputs.toTuple()->elements()[1].toTensor().to(torch::kCPU);		// (1, 1)
			const auto v_accessor = v_tensor.accessor<float, 2>();
			// std::cout << "v_tensor:" << v_tensor << std::endl;
			// std::cout << "v_accessor[0][0]:" << v_accessor[0][0] << std::endl;
			V = v_accessor[0][0];
			for (size_t i = 0; i < num_children; i++) {
				pi[i] = p_accessor[0][i];
			}
			for (size_t hollow : {30,31,32,39,40,41,48,49,50}) {
				pi[hollow] = 0;
			}
			// std::cout << "pi:";
			// for (const auto &p : pi) { std::cout << p << " ";}
			// std::cout << std::endl;
			// std::cout << "V:" << V << std::endl;
		}

		Node* update(float V) {
			if (DEBUG) { std::cout << "\tMCTS_Node update" << std::endl; }
			visit_count += 1;
			Q += (V - Q) / visit_count;
			if (DEBUG) { printf("\t\tnode %p -> node %p5\n", this, this->parent); }
			return this->parent;
		}
		std::map<size_t, size_t> get_visit_distr() {
			std::map<size_t, size_t> visit_distr;
			for (size_t i = 0; i < num_children; i++) {
				auto& child = children[i];
				if (child.visit_count == 0) { continue; }
				visit_distr[i] = child.visit_count;
			}
			return visit_distr;
		}
		void add_dirichlet_noise(size_t size, std::default_random_engine &engine) {
			float noise[81];
      		std::gamma_distribution<float> gamma(0.3f);
			float sum = 0.f;
			for (size_t i = 0; i < size; ++i) {
				noise[i] = gamma(engine);
				sum += noise[i];
			}
			const constexpr float eps = .25f;
			for (size_t i = 0; i < size; ++i) {
				pi[i] = (1 - eps) * pi[i] + eps * (noise[i] / sum);
			}
		}
	};
};
