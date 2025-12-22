//
// Created by mmz on 25-6-13.
//

#ifndef RM_MSGS_H
#define RM_MSGS_H


namespace auto_aim {
    enum class EnemyColor {
        RED = 0,
        BLUE = 1,
        WHITE = 2,
        INVALID = 3,
    };

    struct DebugLight {
        int center_x;
        bool is_light;
        float ratio;
        float angle;
    };

    struct DebugArmor {
        int center_x;
        std::string type;
        float light_ratio;
        float center_distance;
        float angle;
    };
}
#endif //RM_MSGS_H
