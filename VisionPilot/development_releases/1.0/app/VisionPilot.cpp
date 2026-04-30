#include <iostream>
#include <autodrive/autodrive.hpp>
#include <can_reader/can_reader.hpp>
#include <visualization/visualization.hpp>
// #include <inference/autodrive/autodrive.hpp>

int main() {
    std::cout << "Hello and welcome to  VisionPilot!\n";

    can_reader::CanReader reader;
    std::cout << reader.read() << std::endl;

    return 0;
}
