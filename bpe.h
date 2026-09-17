#ifndef BPE_H
#define BPE_H

#include <iosfwd>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Type alias for an adjacent pair of symbols
using SymbolPair = std::pair<std::string, std::string>;

// Byte Pair Encoding tokenizer: learns sub-word merges from a corpus,
// then encodes text into the token ids the model embeds.
class BPETokenizer {
private:
    static const std::string END_OF_WORD; // Marks a word boundary so merges stay inside words
    static const std::string UNK;

    std::vector<SymbolPair> merges;       // Learned merges, in the order they were learned
    std::map<SymbolPair, int> merge_rank; // Merge -> its rank (lower = applied earlier)
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::string> id_to_token;

    // Helper: Register a token, returning its (possibly existing) id
    int add_token(const std::string& token);

    // Helper: Split text on whitespace
    static std::vector<std::string> split_words(const std::string& text);

    // Helper: Explode a word into single-character symbols plus the end-of-word marker
    static std::vector<std::string> to_symbols(const std::string& word);

    // Helper: Replace every occurrence of a symbol pair with the merged symbol
    static std::vector<std::string> apply_merge(const std::vector<std::string>& symbols,
                                                const SymbolPair& pair);

public:
    // Learn merges: repeatedly fuse the most frequent adjacent symbol pair in the corpus
    void train(const std::vector<std::string>& corpus, int num_merges);

    // Tokenize one word by applying the learned merges in rank order
    std::vector<std::string> tokenize_word(const std::string& word) const;

    // Tokenize a whole string into sub-word pieces
    std::vector<std::string> tokenize(const std::string& text) const;

    // Encode a string into token ids (unknown pieces fall back to <unk>)
    std::vector<int> encode(const std::string& text) const;

    // Decode token ids back into text
    std::string decode(const std::vector<int>& ids) const;

    // Persist the learned vocabulary and merges alongside the model weights,
    // so inference tokenizes exactly the way training did
    void save(std::ostream& out) const;
    void load(std::istream& in);

    const std::string& token(int id) const;
    int vocab_size() const;
    int num_merges() const;
};

#endif // BPE_H
