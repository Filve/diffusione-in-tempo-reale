#include "core/pe_imports.h"

#include <algorithm>
#include <cstring>

namespace sd::pe {

    namespace {

        constexpr uint32_t PE_SIGNATURE         = 0x00004550;  // "PE\0\0"
        constexpr uint16_t MAGIC_PE32           = 0x10b;
        constexpr uint16_t MAGIC_PE32_PLUS      = 0x20b;
        constexpr size_t SECTION_HEADER_SIZE    = 40;
        constexpr size_t IMPORT_DESCRIPTOR_SIZE = 20;
        constexpr size_t MAX_SECTIONS           = 96;
        constexpr size_t MAX_IMPORTS            = 1024;
        constexpr size_t MAX_NAME_LENGTH        = 260;

        struct Section {
            uint32_t virtual_address = 0;
            uint32_t raw_size        = 0;
            uint32_t raw_pointer     = 0;
        };

        class Reader {
        public:
            explicit Reader(const ReadAt& read_at)
                : read_at_(read_at) {}

            bool u16(uint64_t offset, uint16_t& value) const {
                uint8_t b[2];
                if (!read_at_(offset, b, sizeof(b))) {
                    return false;
                }
                value = static_cast<uint16_t>(b[0] | (b[1] << 8));
                return true;
            }

            bool u32(uint64_t offset, uint32_t& value) const {
                uint8_t b[4];
                if (!read_at_(offset, b, sizeof(b))) {
                    return false;
                }
                value = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
                        (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
                return true;
            }

            bool bytes(uint64_t offset, void* dst, size_t len) const {
                return read_at_(offset, dst, len);
            }

        private:
            const ReadAt& read_at_;
        };

        bool rva_to_offset(const std::vector<Section>& sections, uint32_t rva, uint64_t& offset) {
            for (const Section& s : sections) {
                if (rva >= s.virtual_address && rva - s.virtual_address < s.raw_size) {
                    offset = static_cast<uint64_t>(s.raw_pointer) + (rva - s.virtual_address);
                    return true;
                }
            }
            return false;
        }

        // A DLL name is plain ASCII; anything else means the data is not an import table.
        bool read_name(const Reader& reader, uint64_t offset, std::string& name) {
            name.clear();
            for (size_t i = 0; i < MAX_NAME_LENGTH; ++i) {
                uint8_t c;
                if (!reader.bytes(offset + i, &c, 1)) {
                    return false;
                }
                if (c == 0) {
                    return !name.empty();
                }
                if (c < 0x20 || c > 0x7e) {
                    return false;
                }
                name.push_back(static_cast<char>(c));
            }
            return false;
        }

    }  // namespace

    std::vector<std::string> imported_dlls(const ReadAt& read_at) {
        std::vector<std::string> dlls;
        const Reader reader(read_at);

        uint8_t mz[2];
        uint32_t pe_offset = 0;
        uint32_t signature = 0;
        if (!reader.bytes(0, mz, sizeof(mz)) || mz[0] != 'M' || mz[1] != 'Z' ||
            !reader.u32(0x3c, pe_offset) || !reader.u32(pe_offset, signature) || signature != PE_SIGNATURE) {
            return dlls;
        }

        // COFF header follows the signature: Machine, NumberOfSections, ..., SizeOfOptionalHeader.
        const uint64_t coff    = static_cast<uint64_t>(pe_offset) + 4;
        uint16_t section_count = 0;
        uint16_t optional_size = 0;
        uint16_t magic         = 0;
        if (!reader.u16(coff + 2, section_count) || !reader.u16(coff + 16, optional_size)) {
            return dlls;
        }
        const uint64_t optional = coff + 20;
        if (!reader.u16(optional, magic)) {
            return dlls;
        }
        uint64_t directories = 0;
        if (magic == MAGIC_PE32_PLUS) {
            directories = optional + 112;
        } else if (magic == MAGIC_PE32) {
            directories = optional + 96;
        } else {
            return dlls;
        }

        // The import directory is data directory #1: (VirtualAddress, Size).
        uint32_t directory_count = 0;
        uint32_t import_rva      = 0;
        if (!reader.u32(directories - 4, directory_count) || directory_count < 2 ||
            !reader.u32(directories + 8, import_rva) || import_rva == 0) {
            return dlls;
        }

        std::vector<Section> sections;
        const uint64_t section_table = optional + optional_size;
        for (size_t i = 0; i < std::min<size_t>(section_count, MAX_SECTIONS); ++i) {
            const uint64_t at = section_table + i * SECTION_HEADER_SIZE;
            Section s;
            if (!reader.u32(at + 12, s.virtual_address) || !reader.u32(at + 16, s.raw_size) ||
                !reader.u32(at + 20, s.raw_pointer)) {
                return dlls;
            }
            sections.push_back(s);
        }

        uint64_t table = 0;
        if (!rva_to_offset(sections, import_rva, table)) {
            return dlls;
        }
        for (size_t i = 0; i < MAX_IMPORTS; ++i) {
            uint8_t descriptor[IMPORT_DESCRIPTOR_SIZE];
            if (!reader.bytes(table + i * IMPORT_DESCRIPTOR_SIZE, descriptor, sizeof(descriptor))) {
                break;
            }
            uint32_t name_rva = static_cast<uint32_t>(descriptor[12]) | (static_cast<uint32_t>(descriptor[13]) << 8) |
                                (static_cast<uint32_t>(descriptor[14]) << 16) | (static_cast<uint32_t>(descriptor[15]) << 24);
            if (name_rva == 0) {
                break;
            }
            uint64_t name_offset = 0;
            std::string name;
            if (!rva_to_offset(sections, name_rva, name_offset) || !read_name(reader, name_offset, name)) {
                break;
            }
            dlls.push_back(name);
        }
        return dlls;
    }

}  // namespace sd::pe
