#include "json.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

const JsonValue* JsonValue::find(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    for (const auto& member : object) {
        if (member.first == key) return &member.second;
    }
    return nullptr;
}

namespace {

// Recursive-descent parser over a cursor into the document
class Parser {
public:
    explicit Parser(const std::string& text) : src(text) {}

    JsonValue parse_document() {
        skip_whitespace();
        JsonValue value = parse_value(0);
        skip_whitespace();
        if (pos != src.size()) {
            fail("trailing content after the top-level value");
        }
        return value;
    }

private:
    const std::string& src;
    size_t pos = 0;

    static const int MAX_DEPTH = 64;

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("JSON at offset " + std::to_string(pos) + ": " + message);
    }

    bool done() const { return pos >= src.size(); }
    char peek() const { return src[pos]; }

    void skip_whitespace() {
        while (!done()) {
            char c = src[pos];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos;
            } else {
                break;
            }
        }
    }

    void expect(char c) {
        if (done() || src[pos] != c) {
            fail(std::string("expected '") + c + "'");
        }
        ++pos;
    }

    bool consume_literal(const char* literal) {
        size_t length = std::string(literal).size();
        if (src.compare(pos, length, literal) == 0) {
            pos += length;
            return true;
        }
        return false;
    }

    // Encode one code point as UTF-8, so \u escapes survive into the output
    static void append_utf8(std::string& out, unsigned code_point) {
        if (code_point < 0x80) {
            out += static_cast<char>(code_point);
        } else if (code_point < 0x800) {
            out += static_cast<char>(0xC0 | (code_point >> 6));
            out += static_cast<char>(0x80 | (code_point & 0x3F));
        } else if (code_point < 0x10000) {
            out += static_cast<char>(0xE0 | (code_point >> 12));
            out += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code_point & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (code_point >> 18));
            out += static_cast<char>(0x80 | ((code_point >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code_point & 0x3F));
        }
    }

    unsigned parse_hex4() {
        if (pos + 4 > src.size()) fail("truncated \\u escape");
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            char c = src[pos++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<unsigned>(c - 'A' + 10);
            else fail("bad hex digit in \\u escape");
        }
        return value;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (done()) fail("unterminated string");
            char c = src[pos++];

            if (c == '"') break;
            if (c != '\\') {
                out += c;
                continue;
            }

            if (done()) fail("unterminated escape");
            char escape = src[pos++];
            switch (escape) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned code_point = parse_hex4();
                    // Recombine a surrogate pair into the code point it encodes
                    if (code_point >= 0xD800 && code_point <= 0xDBFF &&
                        src.compare(pos, 2, "\\u") == 0) {
                        size_t saved = pos;
                        pos += 2;
                        unsigned low = parse_hex4();
                        if (low >= 0xDC00 && low <= 0xDFFF) {
                            code_point = 0x10000 + ((code_point - 0xD800) << 10) + (low - 0xDC00);
                        } else {
                            pos = saved; // Not a pair after all; leave it for the next round
                        }
                    }
                    append_utf8(out, code_point);
                    break;
                }
                default: fail("unknown escape character");
            }
        }
        return out;
    }

    JsonValue parse_number() {
        size_t start = pos;
        if (!done() && (peek() == '-' || peek() == '+')) ++pos;
        while (!done()) {
            char c = peek();
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
                c == '+' || c == '-') {
                ++pos;
            } else {
                break;
            }
        }
        if (start == pos) fail("expected a number");

        JsonValue value;
        value.type = JsonValue::Type::Number;
        value.number = std::strtod(src.substr(start, pos - start).c_str(), nullptr);
        return value;
    }

    JsonValue parse_array(int depth) {
        JsonValue value;
        value.type = JsonValue::Type::Array;

        expect('[');
        skip_whitespace();
        if (!done() && peek() == ']') {
            ++pos;
            return value;
        }

        while (true) {
            skip_whitespace();
            value.array.push_back(parse_value(depth + 1));
            skip_whitespace();
            if (done()) fail("unterminated array");
            if (peek() == ',') {
                ++pos;
                continue;
            }
            if (peek() == ']') {
                ++pos;
                break;
            }
            fail("expected ',' or ']' in array");
        }
        return value;
    }

    JsonValue parse_object(int depth) {
        JsonValue value;
        value.type = JsonValue::Type::Object;

        expect('{');
        skip_whitespace();
        if (!done() && peek() == '}') {
            ++pos;
            return value;
        }

        while (true) {
            skip_whitespace();
            std::string key = parse_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();
            value.object.push_back({key, parse_value(depth + 1)});
            skip_whitespace();
            if (done()) fail("unterminated object");
            if (peek() == ',') {
                ++pos;
                continue;
            }
            if (peek() == '}') {
                ++pos;
                break;
            }
            fail("expected ',' or '}' in object");
        }
        return value;
    }

    JsonValue parse_value(int depth) {
        if (depth > MAX_DEPTH) fail("nested too deeply");
        if (done()) fail("unexpected end of document");

        char c = peek();
        if (c == '{') return parse_object(depth);
        if (c == '[') return parse_array(depth);

        if (c == '"') {
            JsonValue value;
            value.type = JsonValue::Type::String;
            value.string = parse_string();
            return value;
        }

        if (c == 't' || c == 'f') {
            JsonValue value;
            value.type = JsonValue::Type::Bool;
            if (consume_literal("true")) {
                value.boolean = true;
            } else if (consume_literal("false")) {
                value.boolean = false;
            } else {
                fail("expected 'true' or 'false'");
            }
            return value;
        }
        if (consume_literal("null")) {
            return JsonValue();
        }

        return parse_number();
    }
};

} // namespace

JsonValue json_parse(const std::string& text) {
    Parser parser(text);
    return parser.parse_document();
}

JsonValue json_parse_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("could not open '" + path + "'");
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return json_parse(buffer.str());
}
