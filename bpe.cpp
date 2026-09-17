#include "bpe.h"

#include "io.h"

#include <algorithm>
#include <limits>
#include <sstream>

const std::string BPETokenizer::END_OF_WORD = "</w>";
const std::string BPETokenizer::UNK = "<unk>";

int BPETokenizer::add_token(const std::string& token) {
    auto it = token_to_id.find(token);
    if (it != token_to_id.end()) return it->second;

    int id = static_cast<int>(id_to_token.size());
    token_to_id[token] = id;
    id_to_token.push_back(token);
    return id;
}

std::vector<std::string> BPETokenizer::split_words(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream stream(text);
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

std::vector<std::string> BPETokenizer::to_symbols(const std::string& word) {
    std::vector<std::string> symbols;
    for (char c : word) {
        symbols.push_back(std::string(1, c));
    }
    symbols.push_back(END_OF_WORD);
    return symbols;
}

std::vector<std::string> BPETokenizer::apply_merge(const std::vector<std::string>& symbols,
                                                   const SymbolPair& pair) {
    std::vector<std::string> merged;
    for (size_t i = 0; i < symbols.size(); ) {
        if (i + 1 < symbols.size() && symbols[i] == pair.first && symbols[i + 1] == pair.second) {
            merged.push_back(pair.first + pair.second);
            i += 2;
        } else {
            merged.push_back(symbols[i]);
            i += 1;
        }
    }
    return merged;
}

void BPETokenizer::train(const std::vector<std::string>& corpus, int num_merges) {
    // 1. Count each distinct word, then hold it as a sequence of symbols
    std::map<std::string, int> word_counts;
    for (const auto& line : corpus) {
        for (const auto& word : split_words(line)) {
            word_counts[word]++;
        }
    }

    std::vector<std::pair<std::vector<std::string>, int>> words;
    for (const auto& entry : word_counts) {
        words.push_back({to_symbols(entry.first), entry.second});
    }

    // 2. Seed the vocabulary with the unknown token and every base symbol
    add_token(UNK);
    for (const auto& word : words) {
        for (const auto& symbol : word.first) {
            add_token(symbol);
        }
    }

    // 3. Merge the most frequent pair, num_merges times
    for (int step = 0; step < num_merges; ++step) {
        std::map<SymbolPair, int> pair_counts;
        for (const auto& word : words) {
            const auto& symbols = word.first;
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                pair_counts[{symbols[i], symbols[i + 1]}] += word.second;
            }
        }
        if (pair_counts.empty()) break;

        // Ordered map + first-max keeps ties deterministic
        auto best = std::max_element(pair_counts.begin(), pair_counts.end(),
            [](const std::pair<const SymbolPair, int>& a,
               const std::pair<const SymbolPair, int>& b) {
                return a.second < b.second;
            });
        if (best->second < 2) break; // Nothing repeats often enough to be worth merging

        SymbolPair pair = best->first;
        merge_rank[pair] = static_cast<int>(merges.size());
        merges.push_back(pair);
        add_token(pair.first + pair.second);

        for (auto& word : words) {
            word.first = apply_merge(word.first, pair);
        }
    }
}

std::vector<std::string> BPETokenizer::tokenize_word(const std::string& word) const {
    std::vector<std::string> symbols = to_symbols(word);

    while (symbols.size() > 1) {
        // Find the applicable merge that was learned earliest
        int best_rank = std::numeric_limits<int>::max();
        SymbolPair best_pair;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = merge_rank.find({symbols[i], symbols[i + 1]});
            if (it != merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pair = it->first;
            }
        }
        if (best_rank == std::numeric_limits<int>::max()) break; // No merge applies

        symbols = apply_merge(symbols, best_pair);
    }
    return symbols;
}

std::vector<std::string> BPETokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    for (const auto& word : split_words(text)) {
        for (const auto& token : tokenize_word(word)) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    for (const auto& token : tokenize(text)) {
        auto it = token_to_id.find(token);
        ids.push_back(it != token_to_id.end() ? it->second : token_to_id.at(UNK));
    }
    return ids;
}

std::string BPETokenizer::decode(const std::vector<int>& ids) const {
    std::string text;
    for (int id : ids) {
        const std::string& token = id_to_token.at(id);
        text += (token == END_OF_WORD) ? " " : token;
    }
    return text;
}

void BPETokenizer::save(std::ostream& out) const {
    // Vocabulary, in id order
    write_i32(out, static_cast<int32_t>(id_to_token.size()));
    for (const auto& token : id_to_token) {
        write_string(out, token);
    }

    // Merges, in the order they were learned (rank is the index)
    write_i32(out, static_cast<int32_t>(merges.size()));
    for (const auto& pair : merges) {
        write_string(out, pair.first);
        write_string(out, pair.second);
    }
}

void BPETokenizer::load(std::istream& in) {
    id_to_token.clear();
    token_to_id.clear();
    merges.clear();
    merge_rank.clear();

    int32_t vocab_count = read_i32(in);
    id_to_token.reserve(static_cast<size_t>(vocab_count));
    for (int32_t i = 0; i < vocab_count; ++i) {
        std::string token = read_string(in);
        token_to_id[token] = static_cast<int>(id_to_token.size());
        id_to_token.push_back(token);
    }

    int32_t merge_count = read_i32(in);
    merges.reserve(static_cast<size_t>(merge_count));
    for (int32_t i = 0; i < merge_count; ++i) {
        std::string first = read_string(in);
        std::string second = read_string(in);
        merge_rank[{first, second}] = static_cast<int>(merges.size());
        merges.push_back({first, second});
    }
}

const std::string& BPETokenizer::token(int id) const { return id_to_token.at(id); }

int BPETokenizer::vocab_size() const { return static_cast<int>(id_to_token.size()); }

int BPETokenizer::num_merges() const { return static_cast<int>(merges.size()); }
