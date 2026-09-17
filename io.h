#ifndef IO_H
#define IO_H

#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

// Binary read/write primitives shared by the tokenizer and the model checkpoint.
// The layout is host-endian and host float format: checkpoints are meant to move
// between runs on one machine, not between architectures.

inline void write_i32(std::ostream& out, int32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

inline int32_t read_i32(std::istream& in) {
    int32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("checkpoint: unexpected end of file");
    return value;
}

inline void write_floats(std::ostream& out, const std::vector<float>& values) {
    write_i32(out, static_cast<int32_t>(values.size()));
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
}

inline void read_floats(std::istream& in, std::vector<float>& values) {
    int32_t count = read_i32(in);
    if (count < 0) throw std::runtime_error("checkpoint: negative array length");
    values.resize(static_cast<size_t>(count));
    if (count > 0) {
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!in) throw std::runtime_error("checkpoint: truncated float array");
    }
}

inline void write_string(std::ostream& out, const std::string& text) {
    write_i32(out, static_cast<int32_t>(text.size()));
    if (!text.empty()) {
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
    }
}

inline std::string read_string(std::istream& in) {
    int32_t length = read_i32(in);
    if (length < 0) throw std::runtime_error("checkpoint: negative string length");
    std::string text(static_cast<size_t>(length), '\0');
    if (length > 0) {
        in.read(&text[0], length);
        if (!in) throw std::runtime_error("checkpoint: truncated string");
    }
    return text;
}

#endif // IO_H
