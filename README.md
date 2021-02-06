# pyManifold
A library implementing the Lie Groups SO(2), SE(2), SO(3), and SE(3) in Python3. Based off of Transformations for Dummies by James Jackson and A Micro Lie Theory... by Sola.
Note that the quaternion class has the same functionality as SO3 but is more stable.
I have some questions about the jacobians for the Quaternion and SE3 (uses a quaternion internally) which behave a little weird. Don't use the jacobian functionality for these classes
