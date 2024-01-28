#include <plan_manage/trajectory_generator_waypoint.h>
#include <stdio.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

using namespace std;    
using namespace Eigen;

TrajectoryGeneratorWaypoint::TrajectoryGeneratorWaypoint(){

}
TrajectoryGeneratorWaypoint::~TrajectoryGeneratorWaypoint(){
    
}

//define factorial function, input i, output i!
int TrajectoryGeneratorWaypoint::Factorial(int x)
{
    int fac = 1;
    for(int i = x; i > 0; i--)
        fac = fac * i;
    return fac;
}

Eigen::VectorXd TrajectoryGeneratorWaypoint::timeAllocation(Eigen::MatrixXd &Path)
{ 
    const double _Vel = 2.0;
    const double _Acc = 2.0;

    VectorXd time(Path.rows() - 1);

    // The time allocation is many relative timelines but not one common timeline
    for(int i = 0; i < time.rows(); i++)
    {
        double distance = (Path.row(i+1) - Path.row(i)).norm();    // or .lpNorm<2>()
        double x1 = _Vel * _Vel / (2 * _Acc); 
        double x2 = distance - 2 * x1;
        double t1 = _Vel / _Acc;
        double t2 = x2 / _Vel;
        time(i) = 2 * t1 + t2;
    }
    // cout << time << endl;

    return time;
}

Eigen::VectorXd TrajectoryGeneratorWaypoint::getPos(Eigen::VectorXd &coeff, double &cur_t, int &dof)
{
    int co_num = coeff.rows();
    int order = co_num/dof;// poly order
    Eigen::VectorXd position(dof);

    for (int i=0; i<dof; ++i)
    {
        double temp = 0.0;
        for (int j=0; j<order; ++j)
        {
            temp += coeff(i*order+j)*pow(cur_t,order-1-j);
        }
        cout << temp << endl;
        position(i) = temp;
    }
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    cout << position << endl;
    return position;
}

Eigen::VectorXd TrajectoryGeneratorWaypoint::getVel(Eigen::MatrixXd &coeff, double &cur_t, int &dof)
{
    int co_num = coeff.cols();
    int order = co_num/dof;// poly order
    Eigen::VectorXd velocity(dof);

    for (int i=0; i<dof; ++i)
    {
        double temp = 0.0;
        for (int j=0; j<order-1; ++j)
        {
            temp += (order-1-j)*coeff(0,i*order+j)*pow(cur_t,order-2-j);
        }
        velocity(i) = temp;
    }
    return velocity;
}

Eigen::VectorXd TrajectoryGeneratorWaypoint::getAcc(Eigen::MatrixXd &coeff, double &cur_t, int &dof)
{
    int co_num = coeff.cols();
    int order = co_num/dof;// poly order
    Eigen::VectorXd acceleration(dof);

    for (int i=0; i<dof; ++i)
    {
        double temp = 0.0;
        for (int j=0; j<order-2; ++j)
        {
            temp += (order-2-j)*(order-1-j)*coeff(0,i*order+j)*pow(cur_t,order-3-j);
        }
        acceleration(i) = temp;
    }
    return acceleration;
}

Eigen::VectorXd TrajectoryGeneratorWaypoint::getJer(Eigen::MatrixXd &coeff, double &cur_t, int &dof)
{
    int co_num = coeff.cols();
    int order = co_num/dof;// poly order
    Eigen::VectorXd jerk(dof);

    for (int i=0; i<dof; ++i)
    {
        double temp = 0.0;
        for (int j=0; j<order-3; ++j)
        {
            temp += (order-3-j)*(order-2-j)*(order-1-j)*coeff(0,i*order+j)*pow(cur_t,order-4-j);
        }
        jerk(i) = temp;
    }
    return jerk;
}

Eigen::MatrixXd TrajectoryGeneratorWaypoint::PolyQPGeneration(const int d_order,           // the order of derivative
                                                              const Eigen::MatrixXd &Path, // waypoints coordinates (4d)
                                                              const Eigen::MatrixXd &Vel,  // boundary velocity
                                                              const Eigen::MatrixXd &Acc,  // boundary acceleration
                                                              const Eigen::VectorXd &Time) // time allocation in each segment
{
    int p_order = 2 * d_order - 1;      // the order of polynomial
    int p_num1d = p_order + 1;          // the number of Coefficients in each segment
    int seg_num = Time.size();          // the number of segments

    MatrixXd PolyCoeff = MatrixXd::Zero(seg_num, 4 * p_num1d);     // position(x,y,z,yaw), so we need (4 * p_num1d) coefficients

    VectorXd Px(p_num1d * seg_num);     // coefficients in each axis
    VectorXd Py(p_num1d * seg_num);
    VectorXd Pz(p_num1d * seg_num);

    // enforce initial and final position,velocity and accleration, for higher order derivatives, just assume them be 0
    MatrixXd StartState(d_order, 4);
    MatrixXd EndState(d_order, 4);
    StartState.row(0) = Path.row(0);
    StartState.row(1) = Vel.row(0);
    StartState.row(2) = Acc.row(0);
    EndState.row(0) = Path.row((Path.rows()-1));
    EndState.row(1) = Vel.row(1);
    EndState.row(2) = Acc.row(1);
    if(d_order == 4)
    {
        StartState.row(3) = VectorXd::Zero(4);  // jerk
        EndState.row(3) = VectorXd::Zero(4); 
    }
 

    _Q = MatrixXd::Zero(p_num1d * seg_num, p_num1d * seg_num);
    _M = MatrixXd::Zero(p_num1d * seg_num, p_num1d * seg_num);
    _Ct = MatrixXd::Zero(2 * d_order * seg_num, d_order * (seg_num + 1));

    for(int seg_index = 0; seg_index < seg_num; seg_index++)
    {
        // calculate Matrix Q
        _Q.block(seg_index*p_num1d, seg_index*p_num1d, p_num1d, p_num1d) = getQ(p_num1d, d_order, Time, seg_index);
        // calculate Matrix M
        _M.block(seg_index*p_num1d, seg_index*p_num1d, p_num1d, p_num1d) = getM(p_num1d, d_order, Time, seg_index);
    }
    // calculate Matrix Ct
    _Ct = getCt(seg_num, d_order);

    Px = closedFormCalCoeff1D(_Q, _M, _Ct, Path.col(0), StartState.col(0), EndState.col(0), seg_num, d_order);
    Py = closedFormCalCoeff1D(_Q, _M, _Ct, Path.col(1), StartState.col(1), EndState.col(1), seg_num, d_order);
    Pz = closedFormCalCoeff1D(_Q, _M, _Ct, Path.col(2), StartState.col(2), EndState.col(2), seg_num, d_order);

    for(int i = 0; i < seg_num; i++)
    {
        PolyCoeff.row(i).segment(0, p_num1d) = Px.segment(p_num1d*i, p_num1d);
        PolyCoeff.row(i).segment(p_num1d, p_num1d) = Py.segment(p_num1d*i, p_num1d);
        PolyCoeff.row(i).segment(2*p_num1d, p_num1d) = Pz.segment(p_num1d*i, p_num1d);
    }
    // cout << " PolyCoeff = " << endl;
    // cout << PolyCoeff << endl;

    return PolyCoeff;
}

Eigen::MatrixXd TrajectoryGeneratorWaypoint::getQ(const int p_num1d, const int d_order, const Eigen::VectorXd &Time, const int seg_index)
{
    // calculate Matrix Q_k of the seg_index-th segment
    MatrixXd Q_k = MatrixXd::Zero(p_num1d, p_num1d);
    for (int i = 0; i < p_num1d; i++)
    {
        for (int j = 0; j < p_num1d; j++)
        {
            if (i >= p_num1d - d_order && j >= p_num1d - d_order)
            {
                Q_k(i, j) = (Factorial(i) / Factorial(i - d_order)) * ((Factorial(j) / Factorial(j - d_order))) /
                            (i + j - 2 * d_order + 1) * pow(Time(seg_index), (i + j - 2 * d_order + 1)); // Q of one segment
            }
        }
    }
    // cout << " Q_k = " << endl;
    // cout << Q_k << endl;

    return Q_k;
}


Eigen::MatrixXd TrajectoryGeneratorWaypoint::getM(const int p_num1d, const int d_order, const Eigen::VectorXd &Time, const int seg_index)
{
    MatrixXd M_k = MatrixXd::Zero(p_num1d, p_num1d);
    VectorXd t_pow = VectorXd::Zero(p_num1d);
    for(int i = 0; i < p_num1d; i++)
    {
        t_pow(i) = pow(Time(seg_index),i);
    }
    // cout << "t_pow = " << endl;
    // cout << t_pow << endl;

    if(p_num1d == 6)        // minimum jerk
    {
        M_k << 1,     0   ,     0     ,     0     ,      0     ,      0     ,
               0,     1   ,     0     ,     0     ,      0     ,      0     ,
               0,     0   ,     2     ,     0     ,      0     ,      0     ,
               1, t_pow(1),   t_pow(2),   t_pow(3),    t_pow(4),    t_pow(5),
               0,     1   , 2*t_pow(1), 3*t_pow(2),  4*t_pow(3),  5*t_pow(4),
               0,     0   ,     2     , 6*t_pow(1), 12*t_pow(2), 20*t_pow(3);
    }
    else if(p_num1d == 8)   // minimum snap
    {
        M_k << 1,     0   ,     0     ,     0     ,      0     ,      0     ,      0     ,      0     ,
               0,     1   ,     0     ,     0     ,      0     ,      0     ,      0     ,      0     ,
               0,     0   ,     2     ,     0     ,      0     ,      0     ,      0     ,      0     ,
               0,     0   ,     0     ,     6     ,      0     ,      0     ,      0     ,      0     ,
               1, t_pow(1),   t_pow(2),   t_pow(3),    t_pow(4),    t_pow(5),    t_pow(6),    t_pow(7),
               0,     1   , 2*t_pow(1), 3*t_pow(2),  4*t_pow(3),  5*t_pow(4),  6*t_pow(5),  7*t_pow(6),
               0,     0   ,     2     , 6*t_pow(1), 12*t_pow(2), 20*t_pow(3), 30*t_pow(4), 42*t_pow(5),
               0,     0   ,     0     ,     6     , 24*t_pow(1), 60*t_pow(2),120*t_pow(3),210*t_pow(4);
    }
    // cout << "M_k = " << endl;
    // cout << M_k << endl;

    return M_k;
}

Eigen::MatrixXd TrajectoryGeneratorWaypoint::getCt(const int seg_num, const int d_order)
{
    int d_num = 2 * d_order * seg_num;
    int df_and_dp_num = d_order * (seg_num + 1);
    int mid_waypts_num = seg_num - 1;
    int df_num = 2 * d_order + mid_waypts_num;
    // int dp_num = (d_order - 1) * mid_waypts_num;

    Eigen::MatrixXd Ct = MatrixXd::Zero(d_num, df_and_dp_num);
    
    // Ct for the first segment: pos,vel,acc,(jerk)
    Ct.block(0, 0, d_order, d_order) = MatrixXd::Identity(d_order, d_order);
    // Ct for the last segment: pos,vel,acc,(jerk)
    Ct.block(d_num - d_order, df_num - d_order, d_order, d_order) = MatrixXd::Identity(d_order, d_order);

    for(int mid_waypts_index = 0; mid_waypts_index < mid_waypts_num; mid_waypts_index++)
    {
        // Ct for middle waypoints: pos
        Ct(d_order+2*d_order*mid_waypts_index, d_order+mid_waypts_index) = 1;
        Ct(d_order+(d_order+2*d_order*mid_waypts_index), d_order+mid_waypts_index) = 1;

        // Ct for middle waypoints: vel
        Ct(d_order+1+2*d_order*mid_waypts_index, df_num+(d_order-1)*mid_waypts_index) = 1;
        Ct(d_order+(d_order+1+2*d_order*mid_waypts_index), df_num+(d_order-1)*mid_waypts_index) = 1;

        // Ct for middle waypoints: acc
        Ct(d_order+2+2*d_order*mid_waypts_index, (df_num+1)+(d_order-1)*mid_waypts_index) = 1;
        Ct(d_order+(d_order+2+2*d_order*mid_waypts_index), (df_num+1)+(d_order-1)*mid_waypts_index) = 1;

        if(d_order == 4)  // minimum snap
        {
            // Ct for middle waypoints: jerk
            Ct(d_order+3+2*d_order*mid_waypts_index, (df_num+2)+(d_order-1)*mid_waypts_index) = 1;
            Ct(d_order+(d_order+3+2*d_order*mid_waypts_index), (df_num+2)+(d_order-1)*mid_waypts_index) = 1;   
        }
    }
    // cout << "Ct = " << endl;
    // cout << Ct << endl;

    return Ct;
}

Eigen::VectorXd TrajectoryGeneratorWaypoint::closedFormCalCoeff1D(const Eigen::MatrixXd &Q,
                                                                  const Eigen::MatrixXd &M,
                                                                  const Eigen::MatrixXd &Ct,
                                                                  const Eigen::VectorXd &WayPoints1D,
                                                                  const Eigen::VectorXd &StartState1D,
                                                                  const Eigen::VectorXd &EndState1D,
                                                                  const int seg_num,
                                                                  const int d_order)
{
    int df_and_dp_num = d_order * (seg_num + 1);
    int mid_waypts_num = seg_num - 1;
    int df_num = 2 * d_order + mid_waypts_num;
    int dp_num = (d_order - 1) * mid_waypts_num;

    Eigen::MatrixXd C = Ct.transpose();
    Eigen::MatrixXd M_inv = M.inverse();
    Eigen::MatrixXd M_inv_tran = M_inv.transpose();

    Eigen::MatrixXd R = C * M_inv_tran * Q * M_inv * Ct;
    Eigen::MatrixXd R_pp = R.block(df_num, df_num, dp_num, dp_num);
    Eigen::MatrixXd R_fp = R.block(0, df_num, df_num, dp_num);

    // compute dF
    Eigen::VectorXd dF(df_num);
    dF.head(d_order) = StartState1D;    // start state: pos,vel,acc,(jerk)
    dF.segment(d_order, mid_waypts_num) = WayPoints1D.segment(1,WayPoints1D.rows()-2);  // middle waypoints: pos
    dF.tail(d_order) = EndState1D;      // end state: pos,vel,acc,(jerk)
    // cout << "dF = " << endl;
    // cout << dF << endl;
    
    Eigen::VectorXd dP = -R_pp.inverse() * R_fp.transpose() * dF;   // closed-form solution of Unconstrained quadratic programming

    Eigen::VectorXd dF_and_dP(df_and_dp_num);
    dF_and_dP << dF, dP;
    Eigen::VectorXd PolyCoeff1D = M_inv * Ct * dF_and_dP;   // all coefficients of one segment

    return PolyCoeff1D;
}