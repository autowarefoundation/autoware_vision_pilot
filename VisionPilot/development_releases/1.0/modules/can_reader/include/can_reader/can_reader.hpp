//
// Created by atanasko on 27.4.26.
//

#ifndef INC_VISIONPILOT_CAN_READER_H
#define INC_VISIONPILOT_CAN_READER_H
#include <string>

namespace can_reader {
    class CanReader {
    public:
        CanReader() = default;

        ~CanReader() = default;

        // Simple read method
        std::string read();
    };
}

#endif //INC_VISIONPILOT_CAN_READER_H
