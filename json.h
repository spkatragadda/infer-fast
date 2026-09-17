#ifndef JSON_H
#define JSON_H

#include <string>
#include <utility>
#include <vector>

// A minimal JSON reader, enough to load instruction-tuning datasets without
// pulling in a dependency. Parses the whole document into a value tree.
struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object; // Key order preserved

    bool is_null() const { return type == Type::Null; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    // Look up an object member, or nullptr if this is not an object with that key
    const JsonValue* find(const std::string& key) const;
};

// Parse a JSON document. Throws std::runtime_error with a character offset on
// malformed input.
JsonValue json_parse(const std::string& text);

// Read and parse a file in one step
JsonValue json_parse_file(const std::string& path);

#endif // JSON_H
