#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include "board.h"
#include "action.h"
#include "agent.h"
#include "episode.h"
#include "statistic.h"
#include <torch/torch.h>

int main(int argc, const char* argv[]) {
	std::cout << "HollowNoGo-Demo: ";
	std::copy(argv, argv + argc, std::ostream_iterator<const char*>(std::cout, " "));
	std::cout << std::endl << std::endl;

	size_t total = 1000, block = 0, limit = 0;
	std::string black_args, white_args;
	std::string load, save, model_path, sgf_path;
	std::string name = "TCG-HollowNoGo-Demo", version = "2021"; // for GTP shell
	bool summary = false, shell = false;
	for (int i = 1; i < argc; i++) {
		std::string para(argv[i]);
		if (para.find("--total=") == 0) {
			total = std::stoull(para.substr(para.find("=") + 1));
		} else if (para.find("--block=") == 0) {
			block = std::stoull(para.substr(para.find("=") + 1));
		} else if (para.find("--limit=") == 0) {
			limit = std::stoull(para.substr(para.find("=") + 1));
		} else if (para.find("--black=") == 0) {
			black_args = para.substr(para.find("=") + 1);
		} else if (para.find("--white=") == 0) {
			white_args = para.substr(para.find("=") + 1);
		} else if (para.find("--load=") == 0) {
			load = para.substr(para.find("=") + 1);
		} else if (para.find("--save=") == 0) {
			save = para.substr(para.find("=") + 1);
        } else if (para.find("--model_path=") == 0) {
			model_path = para.substr(para.find("=") + 1);
		} else if (para.find("--sgf_path=") == 0) {
			sgf_path = para.substr(para.find("=") + 1);
		} else if (para.find("--name=") == 0) {
			name = para.substr(para.find("=") + 1);
		} else if (para.find("--version=") == 0) {
			version = para.substr(para.find("=") + 1);
		} else if (para.find("--summary") == 0) {
			summary = true;
		}
	}

	statistic stat(total, block, limit);

	if (load.size()) {
		std::ifstream in(load, std::ios::in);
		in >> stat;
		in.close();
		summary |= stat.is_finished();
	}

	AlphaZeroPlayer black("name=black " + black_args + " role=black");
	AlphaZeroPlayer white("name=white " + white_args + " role=white");
    black.load_model(model_path);
    white.load_model(model_path);

    size_t round = 0;
    while (!stat.is_finished()) {
        black.open_episode("~:" + white.name());
        white.open_episode(black.name() + ":~");

        stat.open_episode(black.name() + ":" + white.name());
        episode& game = stat.back();
        while (true) {
            agent& who = game.take_turns(black, white);
            // action move = who.take_action(game.state());
            action move;
            std::map<size_t, size_t> policy_labels;
            std::tie(move, policy_labels) = who.take_action(game.state());
            if (game.apply_action(move, policy_labels) != true) break;
            if (who.check_for_win(game.state())) break;
        }

        agent& win = game.last_turns(black, white);
        stat.close_episode(win.name());
        black.close_episode(win.name());
        white.close_episode(win.name());

        // std::cout << "game:" << game << std::endl << std::endl;
        std::ofstream sgf_file;
        round += 1;
        // sgf_file.open("../sgf/self-play_"+std::to_string(round)+".sgf");
        sgf_file.open(sgf_path+"/self-play_"+std::to_string(round)+".sgf");
        sgf_file << game;
        sgf_file.close();
    }
}