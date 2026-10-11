// Test for the PE import-table reader used to explain Windows backend DLL load failures.
// Build and run (from the repository root; no other library is needed):
//   c++ -std=c++17 -g -fsanitize=address,undefined -Isrc test/pe_imports_test.cpp src/core/pe_imports.cpp -o pe_imports_test
//   ./pe_imports_test
// Exit code 0 = all checks passed.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/pe_imports.h"

static int failures = 0;
#define CHECK(cond, msg)               \
    do {                               \
        if (!(cond)) {                 \
            printf("FAIL: %s\n", msg); \
            failures++;                \
        } else {                       \
            printf("ok:   %s\n", msg); \
        }                              \
    } while (0)

static void put16(std::vector<uint8_t>& image, size_t at, uint16_t v) {
    image[at]     = static_cast<uint8_t>(v);
    image[at + 1] = static_cast<uint8_t>(v >> 8);
}

static void put32(std::vector<uint8_t>& image, size_t at, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        image[at + i] = static_cast<uint8_t>(v >> (8 * i));
    }
}

// A minimal PE image: DOS header, PE header, a .text and a .rdata section; the import
// directory sits in .rdata (RVA 0x2000, file offset 0x600).
static std::vector<uint8_t> make_pe(bool pe32_plus, const std::vector<std::string>& dlls) {
    const size_t pe_offset     = 0x80;
    const size_t optional      = pe_offset + 4 + 20;
    const size_t optional_size = (pe32_plus ? 112 : 96) + 16 * 8;
    const size_t section_table = optional + optional_size;
    const size_t rdata_offset  = 0x600;
    const uint32_t rdata_rva   = 0x2000;
    const size_t descriptors   = (dlls.size() + 1) * 20;
    size_t names_size          = 0;
    for (const auto& d : dlls) {
        names_size += d.size() + 1;
    }

    std::vector<uint8_t> image(rdata_offset + descriptors + names_size + 16, 0);
    image[0] = 'M';
    image[1] = 'Z';
    put32(image, 0x3c, static_cast<uint32_t>(pe_offset));
    memcpy(&image[pe_offset], "PE\0\0", 4);
    put16(image, pe_offset + 4, pe32_plus ? 0x8664 : 0x14c);                 // Machine
    put16(image, pe_offset + 6, 2);                                          // NumberOfSections
    put16(image, pe_offset + 4 + 16, static_cast<uint16_t>(optional_size));  // SizeOfOptionalHeader
    put16(image, optional, pe32_plus ? 0x20b : 0x10b);                       // Magic
    const size_t directories = optional + (pe32_plus ? 112 : 96);
    put32(image, directories - 4, 16);                                   // NumberOfRvaAndSizes
    put32(image, directories + 8, rdata_rva);                            // import directory RVA
    put32(image, directories + 12, static_cast<uint32_t>(descriptors));  // import directory size

    // .text: VA 0x1000, raw 0x200 @ 0x400. .rdata: VA 0x2000, raw 0x200 @ 0x600.
    memcpy(&image[section_table], ".text", 5);
    put32(image, section_table + 8, 0x200);
    put32(image, section_table + 12, 0x1000);
    put32(image, section_table + 16, 0x200);
    put32(image, section_table + 20, 0x400);
    memcpy(&image[section_table + 40], ".rdata", 6);
    put32(image, section_table + 40 + 8, 0x200);
    put32(image, section_table + 40 + 12, rdata_rva);
    put32(image, section_table + 40 + 16, 0x200 + static_cast<uint32_t>(descriptors + names_size));
    put32(image, section_table + 40 + 20, static_cast<uint32_t>(rdata_offset));

    size_t name_at = rdata_offset + descriptors;
    for (size_t i = 0; i < dlls.size(); ++i) {
        put32(image, rdata_offset + i * 20 + 12, rdata_rva + static_cast<uint32_t>(name_at - rdata_offset));
        memcpy(&image[name_at], dlls[i].c_str(), dlls[i].size() + 1);
        name_at += dlls[i].size() + 1;
    }
    return image;
}

static std::vector<std::string> parse(const std::vector<uint8_t>& image) {
    return sd::pe::imported_dlls([&](uint64_t offset, void* dst, size_t len) {
        if (offset > image.size() || len > image.size() - offset) {
            return false;
        }
        memcpy(dst, image.data() + offset, len);
        return true;
    });
}

int main() {
    const std::vector<std::string> dlls = {"KERNEL32.dll", "cudart64_12.dll", "cublas64_12.dll", "VCRUNTIME140.dll"};

    CHECK(parse(make_pe(true, dlls)) == dlls, "PE32+ imports listed in file order");
    CHECK(parse(make_pe(false, dlls)) == dlls, "PE32 imports listed in file order");
    CHECK(parse(make_pe(true, {})).empty(), "image without imports");
    CHECK(parse({}).empty(), "empty file");
    CHECK(parse({'M', 'Z'}).empty(), "DOS stub only");

    std::vector<uint8_t> bad = make_pe(true, dlls);
    bad[0x80]                = 'X';
    CHECK(parse(bad).empty(), "wrong PE signature");
    bad = make_pe(true, dlls);
    put16(bad, 0x80 + 24, 0x1234);
    CHECK(parse(bad).empty(), "unknown optional header magic");
    bad = make_pe(true, {"ok.dll", "bad\x01name.dll"});
    CHECK(parse(bad) == std::vector<std::string>{"ok.dll"}, "non-printable name stops the list");

    // Truncating the file anywhere, or corrupting any byte, must never crash or hang, and
    // every name returned must be one of the real ones or plain printable text.
    const std::vector<uint8_t> good = make_pe(true, dlls);
    bool sane                       = true;
    for (size_t len = 0; len <= good.size(); ++len) {
        std::vector<uint8_t> cut(good.begin(), good.begin() + len);
        for (const auto& name : parse(cut)) {
            sane = sane && !name.empty() && name.size() < 260;
        }
    }
    uint32_t state = 12345;
    for (int round = 0; round < 20000; ++round) {
        std::vector<uint8_t> mutated = good;
        for (int k = 0; k < 4; ++k) {
            state                                  = state * 1664525u + 1013904223u;
            mutated[(state >> 8) % mutated.size()] = static_cast<uint8_t>(state >> 24);
        }
        for (const auto& name : parse(mutated)) {
            for (char c : name) {
                sane = sane && c >= 0x20 && c <= 0x7e;
            }
        }
    }
    CHECK(sane, "truncated and corrupted images are handled safely");

    printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
