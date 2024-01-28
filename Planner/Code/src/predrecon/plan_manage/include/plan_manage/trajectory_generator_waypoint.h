#ifndef _TRAJECTORY_GENERATOR_WAYPOINT_H_
#define _TRAJECTORY_GENERATOR_WAYPOINT_H_

#include <Eigen/Eigen>
#include <vector>

class TrajectoryGeneratorWaypoint
{
private:
    // double _qp_cost;
    Eigen::MatrixXd _Q;
    Eigen::MatrixXd _M;
    Eigen::MatrixXd _Ct;
    
    Eigen::VectorXd _Px, _Py, _Pz;

    Eigen::MatrixXd getQ(const int p_num1d,
                         const int order, 
                         const Eigen::VectorXd &Time, 
                         const int seg_index);

    Eigen::MatrixXd getM(const int p_num1d,
                         const int order, 
                         const Eigen::VectorXd &Time, 
                         const int seg_index);

    Eigen::MatrixXd getCt(const int seg_num, const int d_order);

    Eigen::VectorXd closedFormCalCoeff1D(const Eigen::MatrixXd &Q,
                                         const Eigen::MatrixXd &M,
                                         const Eigen::MatrixXd &Ct,
                                         const Eigen::VectorXd &WayPoints1D,
                                         const Eigen::VectorXd &StartState1D,
                                         const Eigen::VectorXd &EndState1D,
                                         const int seg_num, 
                                         const int d_order);

public:
    TrajectoryGeneratorWaypoint();

    ~TrajectoryGeneratorWaypoint();

    Eigen::MatrixXd PolyQPGeneration(
        const int order,
        const Eigen::MatrixXd &Path,
        const Eigen::MatrixXd &Vel,
        const Eigen::MatrixXd &Acc,
        const Eigen::VectorXd &Time);
    // :return: |x_coeff (8) -- snap|y_coeff (8) -- snap|z_coeff (8) -- snap|yaw_coeff (8) -- snap|
        
    Eigen::VectorXd timeAllocation(Eigen::MatrixXd &Path);

    Eigen::VectorXd getPos(Eigen::VectorXd &coeff, double &cur_t, int &dof);
    Eigen::VectorXd getVel(Eigen::MatrixXd &coeff, double &cur_t, int &dof);
    Eigen::VectorXd getAcc(Eigen::MatrixXd &coeff, double &cur_t, int &dof);
    Eigen::VectorXd getJer(Eigen::MatrixXd &coeff, double &cur_t, int &dof);

    int Factorial(int x);
};


#endif