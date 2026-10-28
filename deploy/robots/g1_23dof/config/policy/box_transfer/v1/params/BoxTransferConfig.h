#pragma once
#include <array>
namespace box_transfer {
inline constexpr int kSensorWidth=76, kFrameWidth=185, kBodyCount=14;
inline constexpr double kDt=0.02;
inline constexpr std::array<int,23> kMotorIds{{0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,22,23,24,25,26}};
inline constexpr std::array<int,5> kLengths{{391,796,391,1001,466}};
inline constexpr std::array<int,5> kOffsets{{0,391,1187,1578,2579}};
inline constexpr std::array<int,4> kGates{{270,220,410,610}};
inline constexpr std::array<const char*,23> kJointNames{{"left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint","waist_yaw_joint","left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint","right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint"}};
inline constexpr std::array<const char*,14> kBodyNames{{"pelvis","left_hip_roll_link","left_knee_link","left_ankle_roll_link","right_hip_roll_link","right_knee_link","right_ankle_roll_link","torso_link","left_shoulder_roll_link","left_elbow_link","left_wrist_roll_rubber_hand","right_shoulder_roll_link","right_elbow_link","right_wrist_roll_rubber_hand"}};
inline constexpr std::array<double,23> kDefault{{-0.312,0.0,0.0,0.669,-0.363,0.0,-0.312,0.0,0.0,0.669,-0.363,0.0,0.0,0.2,0.2,0.0,0.6,0.0,0.2,-0.2,0.0,0.6,0.0}};
inline constexpr std::array<double,23> kScale{{0.5475464629911068,0.35066146637882434,0.5475464629911068,0.35066146637882434,0.43857731392336724,0.43857731392336724,0.5475464629911068,0.35066146637882434,0.5475464629911068,0.35066146637882434,0.43857731392336724,0.43857731392336724,0.5475464629911068,0.3,0.3,0.3,0.3,0.43857731392336724,0.3,0.3,0.3,0.3,0.43857731392336724}};
inline constexpr std::array<double,23> kKp{{40.17923863450712,99.09842777666111,40.17923863450712,99.09842777666111,28.50124619574858,28.50124619574858,40.17923863450712,99.09842777666111,40.17923863450712,99.09842777666111,28.50124619574858,28.50124619574858,40.17923863450712,40.0,40.0,40.0,40.0,14.25062309787429,40.0,40.0,40.0,40.0,14.25062309787429}};
inline constexpr std::array<double,23> kKd{{2.557889775413375,6.308801853496639,2.557889775413375,6.308801853496639,1.814445686584846,1.814445686584846,2.557889775413375,6.308801853496639,2.557889775413375,6.308801853496639,1.814445686584846,1.814445686584846,2.557889775413375,2.0,2.0,2.0,2.0,0.907222843292423,2.0,2.0,2.0,2.0,0.907222843292423}};
inline constexpr std::array<double,23> kEffort{{88.0,139.0,88.0,139.0,50.0,50.0,88.0,139.0,88.0,139.0,50.0,50.0,88.0,25.0,25.0,25.0,25.0,25.0,25.0,25.0,25.0,25.0,25.0}};
inline constexpr std::array<double,23> kJointMin{{-2.5307,-0.5236,-2.7576,-0.087267,-0.87267,-0.2618,-2.5307,-2.9671,-2.7576,-0.087267,-0.87267,-0.2618,-2.618,-3.0892,-1.5882,-2.618,-1.0472,-1.97222,-3.0892,-2.2515,-2.618,-1.0472,-1.97222}};
inline constexpr std::array<double,23> kJointMax{{2.8798,2.9671,2.7576,2.8798,0.5236,0.2618,2.8798,0.5236,2.7576,2.8798,0.5236,0.2618,2.618,2.6704,2.2515,2.618,2.0944,1.97222,2.6704,1.5882,2.618,2.0944,1.97222}};
inline constexpr std::array<double,3> kBoxHalf{{0.14,0.225,0.1625}};
inline constexpr std::array<double,6> kTables{{-1.8369701987210297e-17,-0.2625,0.9,1.774186336855973,-0.4625,0.9}};
inline constexpr std::array<const char*,4> kActorNames{{"navigation","take","carry","put-down"}};
inline constexpr std::array<int,4> kActorWidths{{156,180,180,180}};
inline constexpr std::array<const char*,4> kActorHashes{{"81c3de0e4f00abc614e4fb7e8487e10f6bd84b050ba56aa51b3840cb32361f40","228be8db103ef502d031b7b023ea4ad517a43664707a2ef6ac240337c733beaf","ca618a8125d99f5219c0a5ff67b8e7335e3c5c516fe5550d9a94018d9a9e1d25","e234fd152da4be6ac9f85febcca21d2dcbada2e1990cc7546037716d304c1519"}};
inline constexpr const char* kModelHash="c4edf3eabb36a985f78c5f9ce0128fca3f3abaa6ea40f3eb4cd072e2f6389d41";
inline constexpr const char* kReferenceHash="88aff71b9813954ed35e1798973b0b7c19f070b8d1790d0d24d86d0db3f01e78";
inline constexpr const char* kLayoutHash="c00eb6e503388c22bdbc45a41279b802bb3654c59377c85c03d046d6000b1552";
inline constexpr const char* kSceneHash="397d66cc55881b62bf6de882367523540a0add0fd45dcf51cae9171a7abcab63";
}
