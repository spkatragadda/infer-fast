// Instruction-tune a checkpoint produced by train.cpp on conversation data.
//
//   ./finetune <pretrained.bin> <data.json> <tuned.bin> [options]
//
// The dataset is a JSON array of conversations, each conversation an array of
// messages carrying a role and its content:
//
//   [
//     [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
//     [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
//   ]
//
// Recognized roles are "user", "assistant" and "system". A file that is a flat
// array of message objects is read as a single conversation.
//
// Messages are rendered with plain-text role markers and concatenated, and only
// assistant content carries loss: the model reads the rest but is graded solely
// on what it is supposed to say.

#include "dataset.h"
#include "model.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model_path;
    std::string data_path;
    std::string output_path;
    int epochs = 100;
    float learning_rate = 0.005f; // Gentler than pretraining: the weights already mean something
    float max_grad_norm = 1.0f;
    int max_tokens = 16000;
    unsigned seed = 1234;
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <pretrained.bin> <data.json> <tuned.bin> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --epochs N     passes over the dataset (default 100)\n"
              << "  --lr F         SGD learning rate (default 0.005)\n"
              << "  --clip F       max global gradient norm, 0 disables (default 1.0)\n"
              << "  --max-tokens N longest sequence to train on, 0 disables (default 16000)\n"
              << "  --seed N       shuffle seed (default 1234)\n";
}

bool parse_options(int argc, char** argv, Options& opts) {
    if (argc < 4) return false;
    opts.model_path = argv[1];
    opts.data_path = argv[2];
    opts.output_path = argv[3];

    for (int i = 4; i < argc; ++i) {
        std::string flag = argv[i];
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << flag << "\n";
            return false;
        }
        std::string value = argv[++i];

        if (flag == "--epochs") opts.epochs = std::stoi(value);
        else if (flag == "--lr") opts.learning_rate = std::stof(value);
        else if (flag == "--clip") opts.max_grad_norm = std::stof(value);
        else if (flag == "--max-tokens") opts.max_tokens = std::stoi(value);
        else if (flag == "--seed") opts.seed = static_cast<unsigned>(std::stoul(value));
        else {
            std::cerr << "Unknown option: " << flag << "\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Options opts;
    if (!parse_options(argc, argv, opts)) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        // 1. Load the pretrained weights. The tokenizer rides along with them and
        //    stays frozen: the embedding table is indexed by its ids.
        Model model = Model::load(opts.model_path);
        std::cout << "Loaded " << opts.model_path << ": vocab " << model.config.vocab_size
                  << ", dim " << model.config.embed_dim << ", " << model.config.num_loops
                  << " attention loops" << std::endl;

        // 2. Render the conversations into masked sequences
        DatasetStats stats;
        std::vector<Example> usable = prepare_examples(
            model.tokenizer, build_chat_examples(model.tokenizer, json_parse_file(opts.data_path)),
            opts.max_tokens, stats);

        if (usable.empty()) {
            std::cerr << "No conversation produced a trainable sequence\n";
            return 1;
        }
        print_dataset_stats(stats, opts.max_tokens);

        // 3. Tune, shuffling the order every epoch
        std::cout << "\nTuning " << opts.epochs << " epochs at lr " << opts.learning_rate
                  << ", clip " << opts.max_grad_norm << ":" << std::endl;

        std::vector<size_t> order(usable.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 gen(opts.seed);
        int report_every = std::max(1, opts.epochs / 20);

        for (int epoch = 0; epoch < opts.epochs; ++epoch) {
            std::shuffle(order.begin(), order.end(), gen);

            float epoch_loss = 0.0f;
            for (size_t index : order) {
                const Example& example = usable[index];
                epoch_loss += model.train_step_masked(example.tokens, example.train_on,
                                                      opts.learning_rate, opts.max_grad_norm);
            }

            if (epoch % report_every == 0 || epoch == opts.epochs - 1) {
                std::cout << "  epoch " << epoch << "\tloss "
                          << epoch_loss / static_cast<float>(usable.size()) << std::endl;
            }
        }

        // 4. Save under a new name, leaving the pretrained checkpoint intact
        model.save(opts.output_path);
        std::cout << "\nSaved tuned checkpoint to " << opts.output_path << std::endl;
        std::cout << "Try it with: ./infer " << opts.output_path
                  << " \"User: your question Assistant:\"" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
