/*****************************************************************************
  ** Initialize Manipulator Parameter
  *****************************************************************************/
  addWorld("world",   // world name
          "joint1"); // child name

  addJoint("joint1",  // my name
           "world",   // parent name
           "joint2",  // child name
            math::vector3(0.012, 0.0, 0.017),                // relative position
            math::convertRPYToRotationMatrix(0.0, 0.0, 0.0), // relative orientation
            Z_AXIS,    // axis of rotation
            11,        // actuator id
            M_PI,      // max joint limit (3.14 rad)
            -M_PI,     // min joint limit (-3.14 rad)
            1.0,       // coefficient
            9.8406837e-02,                                                        // mass
            math::inertiaMatrix(3.4543422e-05, -1.6031095e-08, -3.8375155e-07,
                                3.2689329e-05, 2.8511935e-08,
                                1.8850320e-05),                                   // inertial tensor
            math::vector3(-3.0184870e-04, 5.4043684e-04, 0.018 + 2.9433464e-02)   // COM
            );

  addJoint("joint2",  // my name
            "joint1",  // parent name
            "joint3",  // child name
            math::vector3(0.0, 0.0, 0.0595),                // relative position
            math::convertRPYToRotationMatrix(0.0, 0.0, 0.0), // relative orientation
            Y_AXIS,    // axis of rotation
            12,        // actuator id
            M_PI_2,    // max joint limit (1.67 rad)
            -2.05,     // min joint limit (-2.05 rad)
            1.0,       // coefficient
            1.3850917e-01,                                                        // mass
            math::inertiaMatrix(3.3055381e-04, 9.7940978e-08, -3.8505711e-05,
                                3.4290447e-04, -1.5717516e-06,
                                6.0346498e-05),                                   // inertial tensor
            math::vector3(1.0308393e-02, 3.7743363e-04, 1.0170197e-01)            // COM
            );

  addJoint("joint3",  // my name
            "joint2",  // parent name
            "joint4",  // child name
            math::vector3(0.024, 0.0, 0.128),               // relative position
            math::convertRPYToRotationMatrix(0.0, 0.0, 0.0), // relative orientation
            Y_AXIS,    // axis of rotation
            13,        // actuator id
            1.53,      // max joint limit (1.53 rad)
            -M_PI_2,   // min joint limit (-1.67 rad)
            1.0,       // coefficient
            1.3274562e-01,                                                        // mass
            math::inertiaMatrix(3.0654178e-05, -1.2764155e-06, -2.6874417e-07,
                                2.4230292e-04, 1.1559550e-08,
                                2.5155057e-04),                                   // inertial tensor
            math::vector3(9.0909590e-02, 3.8929816e-04, 2.2413279e-04)            // COM
            );

  addJoint("joint4",  // my name
            "joint3",  // parent name
            "gripper", // child name
            math::vector3(0.124, 0.0, 0.0),                 // relative position
            math::convertRPYToRotationMatrix(0.0, 0.0, 0.0), // relative orientation
            Y_AXIS,    // axis of rotation
            14,        // actuator id
            2.0,       // max joint limit (2.0 rad)
            -1.8,      // min joint limit (-1.8 rad)
            1.0,       // coefficient
            1.4327573e-01,                                                        // mass
            math::inertiaMatrix(8.0870749e-05, 0.0, -1.0157896e-06,
                                7.5980465e-05, 0.0,
                                9.3127351e-05),                                   // inertial tensor
            math::vector3(4.4206755e-02, 3.6839985e-07, 8.9142216e-03)            // COM
            );

  addTool("gripper",  // my name
          "joint4",   // parent name
          math::vector3(0.126, 0.0, 0.0),                 // relative position
          math::convertRPYToRotationMatrix(0.0, 0.0, 0.0), // relative orientation
          15,         // actuator id
          0.00,      // max gripper limit (0.00 m)
          -0.02,     // min gripper limit (-0.02 m)
          -0.015,     // Change unit from `meter` to `radian`
          3.2218127e-02 * 2,                                                    // mass
          math::inertiaMatrix(9.5568826e-06, 2.8424644e-06, -3.2829197e-10,
                              2.2552871e-05, -3.1463634e-10,
                              1.7605306e-05),                                   // inertial tensor
          math::vector3(0.028 + 8.3720668e-03, 0.0246, -4.2836895e-07)          // COM
          );
        