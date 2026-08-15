// Copyright 2026 OpenETA contributors
// SPDX-License-Identifier: Apache-2.0

// The M3 grasp mechanism deliberately separates sensing from holding.  The
// Gazebo Contact system is the sole source of grasp evidence; after a valid
// bilateral contact window, this system carries the object kinematically.  It
// never infers contact from poses, meshes, transforms, or distances.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/contacts.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Types.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/Gravity.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/PoseCmd.hh>
#include <gz/transport/Node.hh>
#include <sdf/Collision.hh>
#include <sdf/Element.hh>

namespace openeta::gazebo
{
namespace
{
constexpr std::int64_t kMinimumContactSpanNs = 100'000'000;  // 100 ms
constexpr std::int64_t kFreshContactNs = 250'000'000;        // 250 ms
constexpr std::size_t kMaximumSamples = 512;

struct ContactSample
{
  std::int64_t stampNs{};
  std::string objectLabel;
};

struct WindowVerdict
{
  bool accepted{false};
  std::string objectLabel;
  std::string reason;
};

struct SavedCollision
{
  gz::sim::Entity entity{gz::sim::kNullEntity};
  sdf::Collision element;
};

void SetGravityEnabled(
    gz::sim::EntityComponentManager &_ecm,
    const gz::sim::Entity _link,
    const bool _enabled)
{
  if (_link == gz::sim::kNullEntity)
    return;
  auto *gravity = _ecm.Component<gz::sim::components::GravityEnabled>(_link);
  if (gravity == nullptr)
    _ecm.CreateComponent(
        _link, gz::sim::components::GravityEnabled(_enabled));
  else
    _ecm.SetComponentData<gz::sim::components::GravityEnabled>(_link, _enabled);
}

void SetZeroVelocity(
    gz::sim::EntityComponentManager &_ecm, const gz::sim::Entity _link)
{
  const auto zero = gz::math::Vector3d::Zero;
  auto *linear = _ecm.Component<gz::sim::components::LinearVelocityCmd>(_link);
  if (linear == nullptr)
    _ecm.CreateComponent(_link, gz::sim::components::LinearVelocityCmd(zero));
  else
    _ecm.SetComponentData<gz::sim::components::LinearVelocityCmd>(_link, zero);

  auto *angular = _ecm.Component<gz::sim::components::AngularVelocityCmd>(_link);
  if (angular == nullptr)
    _ecm.CreateComponent(_link, gz::sim::components::AngularVelocityCmd(zero));
  else
    _ecm.SetComponentData<gz::sim::components::AngularVelocityCmd>(_link, zero);
}
}  // namespace

/// \brief A world system implementing M3's bilateral-contact adhesion.
///
/// Contact callbacks only collect native ``gz.msgs.Contacts``.  All ECS
/// mutation happens in PreUpdate, which keeps Gazebo's update thread safe.
class M3AdhesionSystem final
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemPreUpdate
{
 public:
  void Configure(
      const gz::sim::Entity &_entity,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager & /*_ecm*/,
      gz::sim::EventManager & /*_eventMgr*/) override
  {
    this->worldEntity_ = _entity;
    this->robotModelName_ = this->Value(_sdf, "robot_model_name", this->robotModelName_);
    this->mountLinkName_ = this->Value(_sdf, "mount_link_name", this->mountLinkName_);
    this->targetModelName_ = this->Value(_sdf, "target_model_name", this->targetModelName_);
    this->distractorModelName_ =
        this->Value(_sdf, "distractor_model_name", this->distractorModelName_);
    this->leftContactTopic_ = this->Value(_sdf, "left_contact_topic", this->leftContactTopic_);
    this->rightContactTopic_ = this->Value(_sdf, "right_contact_topic", this->rightContactTopic_);

    this->transport_.Subscribe(this->leftContactTopic_,
        &M3AdhesionSystem::OnLeftContacts, this);
    this->transport_.Subscribe(this->rightContactTopic_,
        &M3AdhesionSystem::OnRightContacts, this);

    this->transport_.Advertise(
        "/m3/adhesion/arm_contact_window", &M3AdhesionSystem::OnArm, this);
    this->transport_.Advertise(
        "/m3/adhesion/capture", &M3AdhesionSystem::OnCapture, this);
    this->transport_.Advertise(
        "/m3/adhesion/release", &M3AdhesionSystem::OnRelease, this);
    this->transport_.Advertise(
        "/m3/adhesion/state", &M3AdhesionSystem::OnState, this);
  }

  void PreUpdate(
      const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm) override
  {
    const auto nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        _info.simTime).count();
    std::lock_guard<std::mutex> lock(this->mutex_);

    // A Gazebo reset rewinds simulated time.  Ensure that our private state
    // cannot survive into the next episode with an object left non-colliding.
    if (this->hasSimulationTime_ && nowNs < this->lastSimulationTimeNs_)
    {
      this->RestoreDynamics(_ecm);
      this->ClearCapture("reset");
      this->armed_ = false;
      this->leftSamples_.clear();
      this->rightSamples_.clear();
      this->phase_ = "IDLE";
    }
    this->hasSimulationTime_ = true;
    this->lastSimulationTimeNs_ = nowNs;

    if (this->captureRequested_)
    {
      this->captureRequested_ = false;
      this->Capture(_ecm);
    }
    if (this->releaseRequested_)
    {
      this->releaseRequested_ = false;
      this->Release(_ecm);
    }

    if (this->captured_)
      this->FollowMount(_ecm);
  }

 private:
  template<typename T>
  T Value(const std::shared_ptr<const sdf::Element> &_sdf,
      const std::string &_name, const T &_default) const
  {
    if (!_sdf || !_sdf->HasElement(_name))
      return _default;
    return _sdf->Get<T>(_name, _default).first;
  }

  void OnLeftContacts(const gz::msgs::Contacts &_contacts)
  {
    this->OnContacts(_contacts, this->leftSamples_);
  }

  void OnRightContacts(const gz::msgs::Contacts &_contacts)
  {
    this->OnContacts(_contacts, this->rightSamples_);
  }

  void OnContacts(const gz::msgs::Contacts &_contacts,
      std::deque<ContactSample> &_samples)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (!this->armed_)
      return;

    // The Contact system emits every contact for this sensor in one message.
    // A message which names more than one candidate, or anything not one of
    // the two whitelisted object models, is deliberately recorded as unknown.
    std::string candidate;
    std::int64_t stampNs{};
    for (const auto &contact : _contacts.contact())
    {
      const auto discovered = this->CandidateFor(
          contact.collision1().name(), contact.collision2().name());
      if (candidate.empty())
        candidate = discovered;
      else if (candidate != discovered)
        candidate = "unknown";
      stampNs = std::max(stampNs, this->StampNs(contact));
    }
    if (candidate.empty())
      return;
    if (stampNs == 0)
      stampNs = this->StampNs(_contacts);
    if (stampNs == 0)
      candidate = "unknown";

    // Coalesce multiple callback deliveries for the exact same physical
    // sample.  Three contact points in one sensor output must not fake three
    // independent samples.
    if (!_samples.empty() && _samples.back().stampNs == stampNs)
    {
      if (_samples.back().objectLabel != candidate)
        _samples.back().objectLabel = "unknown";
      return;
    }
    _samples.push_back({stampNs, candidate});
    while (_samples.size() > kMaximumSamples)
      _samples.pop_front();
  }

  std::int64_t StampNs(const gz::msgs::Contact &_contact) const
  {
    return static_cast<std::int64_t>(_contact.header().stamp().sec()) *
               1'000'000'000LL +
           static_cast<std::int64_t>(_contact.header().stamp().nsec());
  }

  std::int64_t StampNs(const gz::msgs::Contacts &_contacts) const
  {
    return static_cast<std::int64_t>(_contacts.header().stamp().sec()) *
               1'000'000'000LL +
           static_cast<std::int64_t>(_contacts.header().stamp().nsec());
  }

  std::string CandidateFor(const std::string &_first,
      const std::string &_second) const
  {
    const auto target = _first.find(this->targetModelName_) != std::string::npos ||
        _second.find(this->targetModelName_) != std::string::npos;
    const auto distractor = _first.find(this->distractorModelName_) != std::string::npos ||
        _second.find(this->distractorModelName_) != std::string::npos;
    if (target == distractor)
      return "unknown";
    return target ? "target" : "distractor";
  }

  WindowVerdict ValidateWindow() const
  {
    WindowVerdict verdict;
    if (!this->armed_)
    {
      verdict.reason = "contact_window_not_armed";
      return verdict;
    }

    const auto validateSide = [&](const std::deque<ContactSample> &_samples,
                                  const char *_side,
                                  std::string &_candidate,
                                  std::string &_reason) -> bool
    {
      if (_samples.size() < 3)
      {
        _reason = std::string(_side) + "_insufficient_samples";
        return false;
      }
      for (const auto &sample : _samples)
      {
        if (sample.objectLabel != "target" && sample.objectLabel != "distractor")
        {
          _reason = std::string(_side) + "_unknown_contact";
          return false;
        }
        if (_candidate.empty())
          _candidate = sample.objectLabel;
        else if (_candidate != sample.objectLabel)
        {
          _reason = std::string(_side) + "_mixed_contact";
          return false;
        }
      }
      const auto span = _samples.back().stampNs - _samples.front().stampNs;
      if (span < kMinimumContactSpanNs)
      {
        _reason = std::string(_side) + "_contact_span_too_short";
        return false;
      }
      const auto age = this->lastSimulationTimeNs_ - _samples.back().stampNs;
      if (age < 0 || age > kFreshContactNs)
      {
        _reason = std::string(_side) + "_stale_contact";
        return false;
      }
      return true;
    };

    std::string leftCandidate;
    std::string rightCandidate;
    if (!validateSide(this->leftSamples_, "left", leftCandidate, verdict.reason) ||
        !validateSide(this->rightSamples_, "right", rightCandidate, verdict.reason))
      return verdict;
    if (leftCandidate != rightCandidate)
    {
      verdict.reason = "bilateral_mixed_object_contact";
      return verdict;
    }
    verdict.accepted = true;
    verdict.objectLabel = leftCandidate;
    verdict.reason = "bilateral_contact_valid";
    return verdict;
  }

  gz::sim::Entity FindModel(
      gz::sim::EntityComponentManager &_ecm, const std::string &_name) const
  {
    return _ecm.EntityByComponents(
        gz::sim::components::Model(), gz::sim::components::Name(_name));
  }

  gz::sim::Entity FindMountLink(gz::sim::EntityComponentManager &_ecm) const
  {
    const auto robot = this->FindModel(_ecm, this->robotModelName_);
    if (robot == gz::sim::kNullEntity)
      return gz::sim::kNullEntity;
    return gz::sim::Model(robot).LinkByName(_ecm, this->mountLinkName_);
  }

  void Capture(gz::sim::EntityComponentManager &_ecm)
  {
    const auto verdict = this->ValidateWindow();
    if (!verdict.accepted)
    {
      this->phase_ = "REJECTED";
      this->reason_ = verdict.reason;
      return;
    }

    const auto modelName = verdict.objectLabel == "target"
        ? this->targetModelName_ : this->distractorModelName_;
    const auto model = this->FindModel(_ecm, modelName);
    const auto mount = this->FindMountLink(_ecm);
    if (model == gz::sim::kNullEntity || mount == gz::sim::kNullEntity)
    {
      this->phase_ = "REJECTED";
      this->reason_ = "attachment_entities_unavailable";
      return;
    }

    const auto mountPose = gz::sim::worldPose(mount, _ecm);
    const auto objectPose = gz::sim::worldPose(model, _ecm);
    if (!mountPose.Pos().IsFinite() || !objectPose.Pos().IsFinite())
    {
      this->phase_ = "REJECTED";
      this->reason_ = "attachment_pose_unavailable";
      return;
    }

    const auto objectLink = gz::sim::Model(model).CanonicalLink(_ecm);
    if (objectLink == gz::sim::kNullEntity)
    {
      this->phase_ = "REJECTED";
      this->reason_ = "attachment_canonical_link_unavailable";
      return;
    }

    this->capturedModel_ = model;
    this->capturedLink_ = objectLink;
    this->mountLink_ = mount;
    this->relativePose_ = mountPose.Inverse() * objectPose;
    this->captured_ = true;
    this->capturedObjectLabel_ = verdict.objectLabel;
    this->capturedModelName_ = modelName;
    ++this->receiptId_;
    this->armed_ = false;
    this->phase_ = "CAPTURED";
    this->reason_ = "bilateral_contact_adhesion_captured";
    this->FollowMount(_ecm);
  }

  void FollowMount(gz::sim::EntityComponentManager &_ecm)
  {
    if (this->capturedModel_ == gz::sim::kNullEntity ||
        this->mountLink_ == gz::sim::kNullEntity)
    {
      this->phase_ = "REJECTED";
      this->ClearCapture("captured_entity_missing");
      return;
    }

    const auto pose = gz::sim::worldPose(this->mountLink_, _ecm) * this->relativePose_;
    auto *poseCommand =
        _ecm.Component<gz::sim::components::WorldPoseCmd>(this->capturedModel_);
    if (poseCommand == nullptr)
      _ecm.CreateComponent(
          this->capturedModel_, gz::sim::components::WorldPoseCmd(pose));
    else
      _ecm.SetComponentData<gz::sim::components::WorldPoseCmd>(
          this->capturedModel_, pose);

    SetGravityEnabled(_ecm, this->capturedLink_, false);
    this->DisableCollisions(_ecm);
    SetZeroVelocity(_ecm, this->capturedLink_);
  }

  void DisableCollisions(gz::sim::EntityComponentManager &_ecm)
  {
    if (this->collisionsSuppressed_)
      return;

    // gz-sim8's vendor API has no collision-enabled command.  Removing the
    // collision marker and its SDF element is the supported ECS transition
    // available to a system; retaining a copy lets release recreate the
    // original collision entities exactly, rather than inventing geometry.
    for (const auto link : gz::sim::Model(this->capturedModel_).Links(_ecm))
    {
      for (const auto collision : gz::sim::Link(link).Collisions(_ecm))
      {
        const auto *element =
            _ecm.Component<gz::sim::components::CollisionElement>(collision);
        if (element == nullptr)
          continue;
        SavedCollision saved;
        saved.entity = collision;
        saved.element = element->Data();
        this->suppressedCollisions_.push_back(std::move(saved));
        _ecm.RemoveComponent<gz::sim::components::CollisionElement>(collision);
        _ecm.RemoveComponent<gz::sim::components::Collision>(collision);
      }
    }
    this->collisionsSuppressed_ = true;
  }

  void RestoreDynamics(gz::sim::EntityComponentManager &_ecm)
  {
    if (this->capturedModel_ == gz::sim::kNullEntity ||
        this->capturedLink_ == gz::sim::kNullEntity)
      return;
    SetGravityEnabled(_ecm, this->capturedLink_, true);
    SetZeroVelocity(_ecm, this->capturedLink_);
    for (const auto &collision : this->suppressedCollisions_)
    {
      if (_ecm.Component<gz::sim::components::CollisionElement>(collision.entity) == nullptr)
      {
        _ecm.CreateComponent(
            collision.entity,
            gz::sim::components::CollisionElement(collision.element));
      }
      if (_ecm.Component<gz::sim::components::Collision>(collision.entity) == nullptr)
        _ecm.CreateComponent(collision.entity, gz::sim::components::Collision());
    }
    this->suppressedCollisions_.clear();
    this->collisionsSuppressed_ = false;
  }

  void ClearCapture(const std::string &_reason)
  {
    this->captured_ = false;
    this->captureRequested_ = false;
    this->releaseRequested_ = false;
    this->capturedModel_ = gz::sim::kNullEntity;
    this->capturedLink_ = gz::sim::kNullEntity;
    this->mountLink_ = gz::sim::kNullEntity;
    this->capturedObjectLabel_.clear();
    this->capturedModelName_.clear();
    this->suppressedCollisions_.clear();
    this->collisionsSuppressed_ = false;
    this->reason_ = _reason;
  }

  void Release(gz::sim::EntityComponentManager &_ecm)
  {
    if (!this->captured_)
    {
      this->phase_ = "RELEASED";
      this->reason_ = "release_idempotent_not_captured";
      return;
    }
    this->RestoreDynamics(_ecm);
    this->ClearCapture("released_zero_velocity");
    this->phase_ = "RELEASED";
  }

  bool OnArm(const gz::msgs::Boolean &_request, gz::msgs::StringMsg &_reply)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (_request.data())
    {
      if (this->captured_)
      {
        this->phase_ = "CAPTURED";
        this->reason_ = "contact_window_not_armed_while_captured";
      }
      else
      {
        ++this->windowId_;
        this->armed_ = true;
        this->leftSamples_.clear();
        this->rightSamples_.clear();
        this->phase_ = "ARMED";
        this->reason_ = "contact_window_armed";
      }
    }
    else
    {
      this->armed_ = false;
      this->leftSamples_.clear();
      this->rightSamples_.clear();
      this->phase_ = this->captured_ ? "CAPTURED" : "IDLE";
      this->reason_ = "contact_window_cancelled";
    }
    _reply.set_data(this->StateJson());
    return true;
  }

  bool OnCapture(const gz::msgs::Boolean &_request, gz::msgs::StringMsg &_reply)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (!_request.data())
    {
      this->phase_ = "REJECTED";
      this->reason_ = "capture_request_false";
    }
    else if (this->captured_)
    {
      this->phase_ = "CAPTURED";
      this->reason_ = "capture_idempotent_already_captured";
    }
    else
    {
      this->captureRequested_ = true;
      this->phase_ = "CAPTURE_PENDING";
      this->reason_ = "capture_queued";
    }
    _reply.set_data(this->StateJson());
    return true;
  }

  bool OnRelease(const gz::msgs::Boolean &_request, gz::msgs::StringMsg &_reply)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (_request.data())
    {
      this->releaseRequested_ = true;
      this->phase_ = "RELEASE_PENDING";
      this->reason_ = "release_queued";
    }
    else
    {
      this->phase_ = this->captured_ ? "CAPTURED" : "IDLE";
      this->reason_ = "release_request_false";
    }
    _reply.set_data(this->StateJson());
    return true;
  }

  bool OnState(const gz::msgs::Boolean & /*_request*/, gz::msgs::StringMsg &_reply)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    _reply.set_data(this->StateJson());
    return true;
  }

  std::string StateJson() const
  {
    // Values are all fixed internal tokens, so JSON escaping is not required.
    std::ostringstream stream;
    stream << "{\"schema\":\"openeta.m3.adhesion.v1\""
           << ",\"phase\":\"" << this->phase_ << '\"'
           << ",\"armed\":" << (this->armed_ ? "true" : "false")
           << ",\"captured\":" << (this->captured_ ? "true" : "false")
           << ",\"object_label\":\"" << this->capturedObjectLabel_ << '\"'
           << ",\"model_name\":\"" << this->capturedModelName_ << '\"'
           << ",\"receipt_id\":" << this->receiptId_
           << ",\"window_id\":" << this->windowId_
           << ",\"reason\":\"" << this->reason_ << "\"}";
    return stream.str();
  }

  gz::sim::Entity worldEntity_{gz::sim::kNullEntity};
  gz::transport::Node transport_;
  std::mutex mutex_;

  std::string robotModelName_{"rm75_robotiq_2f85_pickplace_sim_v1"};
  std::string mountLinkName_{"gripper_mount_link"};
  std::string targetModelName_{"m3_target"};
  std::string distractorModelName_{"m3_distractor"};
  std::string leftContactTopic_{"/m3/contacts/left_pad"};
  std::string rightContactTopic_{"/m3/contacts/right_pad"};

  std::deque<ContactSample> leftSamples_;
  std::deque<ContactSample> rightSamples_;
  bool armed_{false};
  bool captureRequested_{false};
  bool releaseRequested_{false};
  bool captured_{false};
  bool hasSimulationTime_{false};
  std::int64_t lastSimulationTimeNs_{0};
  std::uint64_t receiptId_{0};
  std::uint64_t windowId_{0};
  std::string phase_{"IDLE"};
  std::string reason_{"uninitialized"};
  std::string capturedObjectLabel_;
  std::string capturedModelName_;
  gz::sim::Entity capturedModel_{gz::sim::kNullEntity};
  gz::sim::Entity capturedLink_{gz::sim::kNullEntity};
  gz::sim::Entity mountLink_{gz::sim::kNullEntity};
  bool collisionsSuppressed_{false};
  std::vector<SavedCollision> suppressedCollisions_;
  gz::math::Pose3d relativePose_;
};
}  // namespace openeta::gazebo

GZ_ADD_PLUGIN(
    openeta::gazebo::M3AdhesionSystem,
    gz::sim::System,
    openeta::gazebo::M3AdhesionSystem::ISystemConfigure,
    openeta::gazebo::M3AdhesionSystem::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
    openeta::gazebo::M3AdhesionSystem,
    "openeta::gazebo::M3AdhesionSystem")
