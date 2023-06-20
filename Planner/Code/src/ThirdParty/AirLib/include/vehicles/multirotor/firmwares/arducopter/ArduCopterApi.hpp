// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef msr_airlib_ArduCopterDroneController_hpp
#define msr_airlib_ArduCopterDroneController_hpp

#include "vehicles/multirotor/api/MultirotorApiBase.hpp"
#include "sensors/SensorCollection.hpp"
#include "physics/Environment.hpp"
#include "physics/Kinematics.hpp"
#include "vehicles/multirotor/MultiRotorParams.hpp"
#include "common/Common.hpp"
#include "physics/PhysicsBody.hpp"
#include "common/AirSimSettings.hpp"

// Sensors
#include "sensors/imu/ImuBase.hpp"
#include "sensors/gps/GpsBase.hpp"
#include "sensors/magnetometer/MagnetometerBase.hpp"
#include "sensors/barometer/BarometerBase.hpp"
#include "sensors/lidar/LidarBase.hpp"

#include "UdpSocket.hpp"

#include <sstream>
#include <iostream>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

namespace msr { namespace airlib {

class ArduCopterApi : public MultirotorApiBase {

public:
    ArduCopterApi(const MultiRotorParams* vehicle_params, const AirSimSettings::MavLinkConnectionInfo& connection_info)
        : ip_(connection_info.udp_address), vehicle_params_(vehicle_params)
    {
        connection_info_ = connection_info;
        sensors_ = &getSensors();

        connect(); // Should we try catching exceptions here?
    }

    ~ArduCopterApi()
    {
        closeConnections();
    }


public:
    virtual void resetImplementation() override
    {
        MultirotorApiBase::resetImplementation();

        // Reset state
    }

    // Update sensor data & send to Ardupilot
    virtual void update() override
    {
        MultirotorApiBase::update();

        sendSensors();
        recvRotorControl();
    }

    // TODO:VehicleApiBase implementation
    virtual bool isApiControlEnabled() const override
    {
        Utils::log("Not Implemented: isApiControlEnabled", Utils::kLogLevelInfo);
        return false;
    }
    virtual void enableApiControl(bool is_enabled) override
    {
        Utils::log("Not Implemented: enableApiControl", Utils::kLogLevelInfo);
        unused(is_enabled);
    }
    virtual bool armDisarm(bool arm) override
    {
        Utils::log("Not Implemented: armDisarm", Utils::kLogLevelInfo);
        unused(arm);
        return false;
    }
    virtual GeoPoint getHomeGeoPoint() const override
    {
        Utils::log("Not Implemented: getHomeGeoPoint", Utils::kLogLevelInfo);
        return GeoPoint(Utils::nan<double>(), Utils::nan<double>(), Utils::nan<float>());
    }
    virtual void getStatusMessages(std::vector<std::string>& messages) override
    {
        unused(messages);
    }

    virtual const SensorCollection& getSensors() const override
    {
        return vehicle_params_->getSensors();
    }

public: //TODO:MultirotorApiBase implementation
    virtual real_T getActuation(unsigned int rotor_index) const override
    {
        return rotor_controls_[rotor_index];
    }

    virtual size_t getActuatorCount() const override
    {
        return vehicle_params_->getParams().rotor_count;
    }

    virtual void moveByRC(const RCData& rc_data) override
    {
        setRCData(rc_data);
    }

    virtual void setSimulatedGroundTruth(const Kinematics::State* kinematics, const Environment* environment) override
    {
        Utils::log("Not Implemented: setSimulatedGroundTruth", Utils::kLogLevelInfo);
        unused(kinematics);
        unused(environment);
    }

    virtual bool setRCData(const RCData& rc_data) override
    {
        last_rcData_ = rc_data;
        is_rc_connected_ = true;

        return true;
    }

protected:
    virtual Kinematics::State getKinematicsEstimated() const override
    {
        Utils::log("Not Implemented: getKinematicsEstimated", Utils::kLogLevelInfo);
        Kinematics::State state;
        return state;
    }

    virtual Vector3r getPosition() const override
    {
        Utils::log("Not Implemented: getPosition", Utils::kLogLevelInfo);
        return Vector3r(Utils::nan<float>(), Utils::nan<float>(), Utils::nan<float>());
    }

    virtual Vector3r getVelocity() const override
    {
        Utils::log("Not Implemented: getVelocity", Utils::kLogLevelInfo);
        return Vector3r(Utils::nan<float>(), Utils::nan<float>(), Utils::nan<float>());
    }

    virtual Quaternionr getOrientation() const override
    {
        Utils::log("Not Implemented: getOrientation", Utils::kLogLevelInfo);
        return Quaternionr(Utils::nan<float>(), Utils::nan<float>(), Utils::nan<float>(), Utils::nan<float>());
    }

    virtual LandedState getLandedState() const override
    {
        Utils::log("Not Implemented: getLandedState", Utils::kLogLevelInfo);
        return LandedState::Landed;
    }

    virtual RCData getRCData() const override
    {
        //return what we received last time through setRCData
        return last_rcData_;
    }

    virtual GeoPoint getGpsLocation() const override
    {
        Utils::log("Not Implemented: getGpsLocation", Utils::kLogLevelInfo);
        return GeoPoint(Utils::nan<double>(), Utils::nan<double>(), Utils::nan<float>());
    }

    virtual float getCommandPeriod() const override
    {
        return 1.0f / 50; //50hz
    }

    virtual float getTakeoffZ() const override
    {
        // pick a number, 3 meters is probably safe
        // enough to get out of the backwash turbulence.  Negative due to NED coordinate system.
        // return params_.takeoff.takeoff_z;
        return 3.0;
    }

    virtual float getDistanceAccuracy() const override
    {
        return 0.5f;    //measured in simulator by firing commands "MoveToLocation -x 0 -y 0" multiple times and looking at distance traveled
    }

    virtual void setControllerGains(uint8_t controllerType, const vector<float>& kp, const vector<float>& ki, const vector<float>& kd) override
    {
        unused(controllerType);
        unused(kp);
        unused(ki);
        unused(kd);
        Utils::log("Not Implemented: setControllerGains", Utils::kLogLevelInfo);
    }

    virtual void commandMotorPWMs(float front_right_pwm, float front_left_pwm, float rear_right_pwm, float rear_left_pwm) override
    {
        unused(front_right_pwm);
        unused(front_left_pwm);
        unused(rear_right_pwm);
        unused(rear_left_pwm);
        Utils::log("Not Implemented: commandMotorPWMs", Utils::kLogLevelInfo);
    }

    virtual void commandRollPitchYawrateThrottle(float roll, float pitch, float yaw_rate, float throttle) override
    {
        unused(roll);
        unused(pitch);
        unused(yaw_rate);
        unused(throttle);
        Utils::log("Not Implemented: commandRollPitchYawrateThrottle", Utils::kLogLevelInfo);
    }

    virtual void commandRollPitchYawZ(float roll, float pitch, float yaw, float z) override
    {
        unused(roll);
        unused(pitch);
        unused(yaw);
        unused(z);
        Utils::log("Not Implemented: commandRollPitchYawZ", Utils::kLogLevelInfo);
    }

    virtual void commandRollPitchYawThrottle(float roll, float pitch, float yaw, float throttle) override
    {
        unused(roll);
        unused(pitch);
        unused(yaw);
        unused(throttle);
        Utils::log("Not Implemented: commandRollPitchYawThrottle", Utils::kLogLevelInfo);
    }

    virtual void commandRollPitchYawrateZ(float roll, float pitch, float yaw_rate, float z) override
    {
        unused(roll);
        unused(pitch);
        unused(yaw_rate);
        unused(z);
        Utils::log("Not Implemented: commandRollPitchYawrateZ", Utils::kLogLevelInfo);
    }

    virtual void commandAngleRatesZ(float roll_rate, float pitch_rate, float yaw_rate, float z) override
    {
        unused(roll_rate);
        unused(pitch_rate);
        unused(yaw_rate);
        unused(z);
        Utils::log("Not Implemented: commandAngleRatesZ", Utils::kLogLevelInfo);
    }

    virtual void commandAngleRatesThrottle(float roll_rate, float pitch_rate, float yaw_rate, float throttle) override
    {
        unused(roll_rate);
        unused(pitch_rate);
        unused(yaw_rate);
        unused(throttle);
        Utils::log("Not Implemented: commandAngleRatesZ", Utils::kLogLevelInfo);
    }

    virtual void commandVelocity(float vx, float vy, float vz, const YawMode& yaw_mode) override
    {
        unused(vx);
        unused(vy);
        unused(vz);
        unused(yaw_mode);
        Utils::log("Not Implemented: commandVelocity", Utils::kLogLevelInfo);
    }

    virtual void commandVelocityZ(float vx, float vy, float z, const YawMode& yaw_mode) override
    {
        unused(vx);
        unused(vy);
        unused(z);
        unused(yaw_mode);
        Utils::log("Not Implemented: commandVelocityZ", Utils::kLogLevelInfo);
    }

    virtual void commandPosition(float x, float y, float z, const YawMode& yaw_mode) override
    {
        unused(x);
        unused(y);
        unused(z);
        unused(yaw_mode);
        Utils::log("Not Implemented: commandPosition", Utils::kLogLevelInfo);
    }

    virtual const MultirotorApiParams& getMultirotorApiParams() const override
    {
        return safety_params_;
    }

    //*** End: MultirotorApiBase implementation ***//

protected:
    void closeConnections()
    {
        if (udpSocket_ != nullptr)
            udpSocket_->close();
    }

    void connect()
    {
        port_ = static_cast<uint16_t>(connection_info_.udp_port);

        closeConnections();

        if (ip_ == "") {
            throw std::invalid_argument("UdpIp setting is invalid.");
        }

        if (port_ == 0) {
            throw std::invalid_argument("UdpPort setting has an invalid value.");
        }

        Utils::log(Utils::stringf("Using UDP port %d, local IP %s, remote IP %s for sending sensor data", port_, connection_info_.local_host_ip.c_str(), ip_.c_str()), Utils::kLogLevelInfo);
        Utils::log(Utils::stringf("Using UDP port %d for receiving rotor power", connection_info_.control_port, connection_info_.local_host_ip.c_str(), ip_.c_str()), Utils::kLogLevelInfo);

        udpSocket_ = std::make_shared<mavlinkcom::UdpSocket>();
        udpSocket_->bind(connection_info_.local_host_ip, connection_info_.control_port);
    }

private:
    virtual void normalizeRotorControls()
    {
        // change 1000-2000 to 0-1.
        for (size_t i = 0; i < Utils::length(rotor_controls_); ++i) {
            rotor_controls_[i] = (rotor_controls_[i] - 1000.0f) / 1000.0f;
        }
    }

    void sendSensors()
    {
        if (sensors_ == nullptr)
            return;

        const auto& gps_output = getGpsData("");
        const auto& imu_output = getImuData("");

        std::ostringstream oss;

        // Send RC channels to Ardupilot if present
        if (is_rc_connected_ && last_rcData_.is_valid) {
            oss << ","
                   "\"rc\": {"
                   "\"channels\": ["
                << (last_rcData_.roll + 1) * 0.5f << ","
                << (last_rcData_.yaw + 1) * 0.5f << ","
                << (last_rcData_.throttle + 1) * 0.5f << ","
                << (-last_rcData_.pitch + 1) * 0.5f << ","
                << static_cast<float>(last_rcData_.getSwitch(0)) << ","
                << static_cast<float>(last_rcData_.getSwitch(1)) << ","
                << static_cast<float>(last_rcData_.getSwitch(2)) << ","
                << static_cast<float>(last_rcData_.getSwitch(3))
                << "]}";
        }

        const uint count_lidars = getSensors().size(SensorBase::SensorType::Lidar);
        // TODO: Add bool value in settings to check whether to send lidar data or not
        // Since it's possible that we don't want to send the lidar data to Ardupilot but still have the lidar (maybe as a ROS topic)
        if (count_lidars != 0) {
            const auto& lidar_output = getLidarData("");
            oss << ","
                   "\"lidar\": {"
                   "\"point_cloud\": [";

            std::copy(lidar_output.point_cloud.begin(), lidar_output.point_cloud.end(), std::ostream_iterator<real_T>(oss, ","));
            oss << "]}";
        }

        float yaw;
        float pitch;
        float roll;
        VectorMath::toEulerianAngle(imu_output.orientation, pitch, roll, yaw);

        char buf[65000];

        // TODO: Split the following sensor packet formation into different parts for individual sensors

        // UDP packets have a maximum size limit of 65kB
        int ret = snprintf(buf, sizeof(buf),
                           "{"
                           "\"timestamp\": %" PRIu64 ","
                           "\"imu\": {"
                           "\"angular_velocity\": [%.12f, %.12f, %.12f],"
                           "\"linear_acceleration\": [%.12f, %.12f, %.12f]"
                           "},"
                           "\"gps\": {"
                           "\"lat\": %.7f,"
                           "\"lon\": %.7f,"
                           "\"alt\": %.3f"
                           "},"
                           "\"velocity\": {"
                           "\"world_linear_velocity\": [%.12f, %.12f, %.12f]"
                           "},"
                           "\"pose\": {"
                           "\"roll\": %.12f,"
                           "\"pitch\": %.12f,"
                           "\"yaw\": %.12f"
                           "}"
                           "%s"
                           "}\n",
                           static_cast<uint64_t>(msr::airlib::ClockFactory::get()->nowNanos() / 1.0E3),
                           imu_output.angular_velocity[0],
                           imu_output.angular_velocity[1],
                           imu_output.angular_velocity[2],
                           imu_output.linear_acceleration[0],
                           imu_output.linear_acceleration[1],
                           imu_output.linear_acceleration[2],
                           gps_output.gnss.geo_point.latitude,
                           gps_output.gnss.geo_point.longitude,
                           gps_output.gnss.geo_point.altitude,
                           gps_output.gnss.velocity[0],
                           gps_output.gnss.velocity[1],
                           gps_output.gnss.velocity[2],
                           roll, pitch, yaw,
                           oss.str().c_str());

        if (ret < 0) {
            Utils::log("Encoding error while forming sensor message", Utils::kLogLevelInfo);
            return;
        }
        else if (static_cast<uint>(ret) >= sizeof(buf)) {
            Utils::log(Utils::stringf("Sensor message truncated, lost %d bytes", ret - sizeof(buf)), Utils::kLogLevelInfo);
        }

        // Send data
        if (udpSocket_ != nullptr) {
            udpSocket_->sendto(buf, strlen(buf), ip_, port_);
        }
    }

    void recvRotorControl()
    {
        // Receive motor data
        RotorControlMessage pkt;
        int recv_ret = udpSocket_->recv(&pkt, sizeof(pkt), 100);
        while (recv_ret != sizeof(pkt)) {
            if (recv_ret <= 0) {
                Utils::log(Utils::stringf("Error while receiving rotor control data - ErrorNo: %d", recv_ret), Utils::kLogLevelInfo);
            } else {
                Utils::log(Utils::stringf("Received %d bytes instead of %zu bytes", recv_ret, sizeof(pkt)), Utils::kLogLevelInfo);
            }

            recv_ret = udpSocket_->recv(&pkt, sizeof(pkt), 100);
        }

        for (auto i = 0; i < kArduCopterRotorControlCount; ++i) {
            rotor_controls_[i] = pkt.pwm[i];
        }

        normalizeRotorControls();
    }

private:
    static const int kArduCopterRotorControlCount = 11;

    struct RotorControlMessage {
        uint16_t pwm[kArduCopterRotorControlCount];
    };

    std::shared_ptr<mavlinkcom::UdpSocket> udpSocket_;

    AirSimSettings::MavLinkConnectionInfo connection_info_;
    uint16_t port_;
    const std::string& ip_;
    const SensorCollection* sensors_;
    const MultiRotorParams* vehicle_params_;

    MultirotorApiParams safety_params_;

    RCData last_rcData_;
    bool is_rc_connected_;

    float rotor_controls_[kArduCopterRotorControlCount];
};

}} //namespace
#endif
