// Copyright 2026 OpenETA contributors
// SPDX-License-Identifier: Apache-2.0

// Authoritative collision-state bridge for a Gazebo DetachableJoint grasp.
//
// Gazebo's fixed joint makes the selected object part of the transported
// mechanism, but it does not automatically suppress contacts between the
// fixed child and the robot.  Those internal contacts over-constrain DART and
// can push the arm outside its unchanged controller path tolerance.  This
// system follows the *actual* DetachableJoint component and changes only the
// selected target's collision-filter mask:
//
//   detached: 0xffff (collides with robot bit 0 and the world)
//   attached: 0x0002 (does not collide with robot bit 0; still collides with
//                     normal world shapes whose mask is 0xffff)
//
// The geometry is never approximated or disabled globally.  Exact sdf::
// Collision objects are removed and recreated beneath the same link because
// Gazebo Sim 8.11 has no runtime CollideBitmaskCmd component.  An ACK is
// published only after the replacement entities are stable in the ECM.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/empty.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/SdfEntityCreator.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/transport/Node.hh>

#include <sdf/Collision.hh>
#include <sdf/Element.hh>
#include <sdf/Surface.hh>

namespace openeta::gazebo
{
class AttachedCollisionFilter final
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemPreUpdate
{
  private: enum class Phase
  {
    kStable,
    kRemoving,
    kSettling,
    kFailed,
  };

  public: void Configure(
      const gz::sim::Entity &,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &_ecm,
      gz::sim::EventManager &_eventManager) override
  {
    if (_sdf)
    {
      if (_sdf->HasElement("target_model"))
        this->targetModel = _sdf->Get<std::string>("target_model");
      if (_sdf->HasElement("target_link"))
        this->targetLinkName = _sdf->Get<std::string>("target_link");
      if (_sdf->HasElement("state_topic"))
        this->stateTopic = _sdf->Get<std::string>("state_topic");
      if (_sdf->HasElement("state_request_topic"))
        this->stateRequestTopic =
            _sdf->Get<std::string>("state_request_topic");
      if (_sdf->HasElement("state_ack_topic"))
        this->stateAckTopic = _sdf->Get<std::string>("state_ack_topic");
      if (_sdf->HasElement("detached_mask"))
        this->detachedMask = static_cast<uint16_t>(
            _sdf->Get<unsigned int>("detached_mask"));
      if (_sdf->HasElement("attached_mask"))
        this->attachedMask = static_cast<uint16_t>(
            _sdf->Get<unsigned int>("attached_mask"));
      if (_sdf->HasElement("robot_mask"))
        this->robotMask = static_cast<uint16_t>(
            _sdf->Get<unsigned int>("robot_mask"));
    }

    if (this->targetModel.empty() || this->targetLinkName.empty() ||
        this->stateTopic.empty() || this->stateRequestTopic.empty() ||
        this->stateAckTopic.empty() ||
        this->robotMask == 0u ||
        this->attachedMask == 0u ||
        (this->robotMask & this->detachedMask) == 0u ||
        (this->robotMask & this->attachedMask) != 0u)
    {
      this->Fail("invalid collision-filter contract");
      return;
    }

    this->creator =
        std::make_unique<gz::sim::SdfEntityCreator>(_ecm, _eventManager);
    this->publisher =
        this->node.Advertise<gz::msgs::Boolean>(this->stateTopic);
    this->ackPublisher =
        this->node.Advertise<gz::msgs::Boolean>(this->stateAckTopic);
    if (!this->node.Subscribe(
            this->stateRequestTopic,
            &AttachedCollisionFilter::OnStateRequest,
            this))
    {
      this->Fail("failed to subscribe to collision-filter state requests");
      return;
    }

    // The compiler guarantees that the initial world contains the full
    // detached mask. World systems are configured before all child model
    // entities exist, so publish that immutable initial state now and resolve
    // the exact target collision templates on the first unpaused PreUpdate.
    this->currentAttached = false;
    this->PublishState();
  }

  public: void PreUpdate(
      const gz::sim::UpdateInfo &,
      gz::sim::EntityComponentManager &_ecm) override
  {
    if (this->phase == Phase::kFailed || !this->creator)
      return;
    if (this->targetLink == gz::sim::kNullEntity && !this->ResolveTarget(_ecm))
      return;

    const bool desiredAttached = this->IsAttached(_ecm);
    this->desiredAttached = desiredAttached;

    if (this->phase == Phase::kStable)
    {
      if (!this->currentAttached.has_value() ||
          this->currentAttached.value() != desiredAttached)
      {
        this->BeginReplacement();
        return;
      }
      this->PublishState();
      return;
    }

    if (this->phase == Phase::kRemoving)
    {
      const auto remaining = _ecm.ChildrenByComponents(
          this->targetLink, gz::sim::components::Collision());
      if (!remaining.empty())
        return;
      this->CreateReplacement(this->transitionAttached, _ecm);
      return;
    }

    if (this->phase == Phase::kSettling)
    {
      const auto collisions = _ecm.ChildrenByComponents(
          this->targetLink, gz::sim::components::Collision());
      if (collisions.size() != this->templates.size())
        return;
      ++this->settlingTicks;
      if (this->settlingTicks < 2u)
        return;
      this->activeCollisions = collisions;
      this->currentAttached = this->transitionAttached;
      this->phase = Phase::kStable;
      this->PublishState();
    }
  }

  private: bool ResolveTarget(gz::sim::EntityComponentManager &_ecm)
  {
    const auto models = _ecm.EntitiesByComponents(
        gz::sim::components::Model(),
        gz::sim::components::Name(this->targetModel));
    if (models.empty())
      return false;
    if (models.size() != 1u)
    {
      this->Fail("target model is not unique");
      return false;
    }
    const auto links = _ecm.ChildrenByComponents(
        models.front(), gz::sim::components::Link(),
        gz::sim::components::Name(this->targetLinkName));
    if (links.empty())
      return false;
    if (links.size() != 1u)
    {
      this->Fail("target link is not unique");
      return false;
    }
    const auto link = links.front();

    std::vector<std::pair<std::string, gz::sim::Entity>> namedCollisions;
    for (const auto entity : _ecm.ChildrenByComponents(
             link, gz::sim::components::Collision()))
    {
      const auto *name = _ecm.Component<gz::sim::components::Name>(entity);
      const auto *element =
          _ecm.Component<gz::sim::components::CollisionElement>(entity);
      if (!name || !element)
      {
        this->Fail("target collision has incomplete ECM provenance");
        return false;
      }
      namedCollisions.emplace_back(name->Data(), entity);
    }
    if (namedCollisions.empty())
      return false;
    std::sort(namedCollisions.begin(), namedCollisions.end(),
        [](const auto &_left, const auto &_right)
        {
          return _left.first < _right.first;
        });

    std::vector<sdf::Collision> templates;
    std::vector<gz::sim::Entity> active;
    for (const auto &[name, entity] : namedCollisions)
    {
      const auto *element =
          _ecm.Component<gz::sim::components::CollisionElement>(entity);
      sdf::Collision collision = element->Data();
      if (collision.Name() != name || !collision.Surface() ||
          !collision.Surface()->Contact() ||
          collision.Surface()->Contact()->CollideBitmask() !=
              this->detachedMask)
      {
        this->Fail("target collision mask or identity is not authoritative");
        return false;
      }
      templates.push_back(std::move(collision));
      active.push_back(entity);
    }
    this->targetLink = link;
    this->templates = std::move(templates);
    this->activeCollisions = std::move(active);
    return true;
  }

  private: bool IsAttached(const gz::sim::EntityComponentManager &_ecm) const
  {
    bool attached = false;
    _ecm.Each<gz::sim::components::DetachableJoint>(
        [&](const gz::sim::Entity &,
            const gz::sim::components::DetachableJoint *_joint)
        {
          if (_joint && _joint->Data().childLink == this->targetLink)
          {
            attached = true;
            return false;
          }
          return true;
        });
    return attached;
  }

  private: void BeginReplacement()
  {
    this->transitionAttached = this->desiredAttached;
    for (const auto entity : this->activeCollisions)
      this->creator->RequestRemoveEntity(entity);
    this->activeCollisions.clear();
    this->phase = Phase::kRemoving;
  }

  private: void CreateReplacement(
      const bool _attached, gz::sim::EntityComponentManager &_ecm)
  {
    const uint16_t mask = _attached ? this->attachedMask : this->detachedMask;
    this->activeCollisions.clear();
    for (const auto &source : this->templates)
    {
      sdf::Collision collision = source;
      sdf::Surface surface;
      if (source.Surface())
        surface = *source.Surface();
      sdf::Contact contact;
      if (source.Surface() && source.Surface()->Contact())
        contact = *source.Surface()->Contact();
      contact.SetCollideBitmask(mask);
      surface.SetContact(contact);
      collision.SetSurface(surface);
      const auto entity = this->creator->CreateEntities(&collision);
      if (entity == gz::sim::kNullEntity)
      {
        this->Fail("failed to recreate target collision");
        return;
      }
      this->creator->SetParent(entity, this->targetLink);
      this->activeCollisions.push_back(entity);
    }
    this->settlingTicks = 0u;
    this->phase = Phase::kSettling;
    (void)_ecm;
  }

  private: void PublishState()
  {
    if (!this->currentAttached.has_value())
      return;
    gz::msgs::Boolean message;
    message.set_data(this->currentAttached.value());
    this->stableState.store(
        this->currentAttached.value() ? 1 : 0,
        std::memory_order_release);
    this->publisher.Publish(message);
  }

  private: void OnStateRequest(const gz::msgs::Empty &)
  {
    const int state = this->stableState.load(std::memory_order_acquire);
    if (state < 0)
      return;
    gz::msgs::Boolean response;
    response.set_data(state == 1);
    this->ackPublisher.Publish(response);
  }

  private: void Fail(const std::string &_detail)
  {
    std::cerr << "[openeta_attached_collision_filter] " << _detail
              << std::endl;
    this->phase = Phase::kFailed;
    this->currentAttached.reset();
  }

  private: std::string targetModel{"target_object"};
  private: std::string targetLinkName{"target_link"};
  private: std::string stateTopic{
      "/openeta/native_grasp/detachable_joint/target/collision_filter_state"};
  private: std::string stateRequestTopic{
      "/openeta/native_grasp/detachable_joint/target/"
      "collision_filter_state/request"};
  private: std::string stateAckTopic{
      "/openeta/native_grasp/detachable_joint/target/"
      "collision_filter_state/ack"};
  private: uint16_t robotMask{0x0001u};
  private: uint16_t detachedMask{0xffffu};
  private: uint16_t attachedMask{0x0002u};
  private: gz::sim::Entity targetLink{gz::sim::kNullEntity};
  private: std::unique_ptr<gz::sim::SdfEntityCreator> creator;
  private: gz::transport::Node node;
  private: gz::transport::Node::Publisher publisher;
  private: gz::transport::Node::Publisher ackPublisher;
  private: std::vector<sdf::Collision> templates;
  private: std::vector<gz::sim::Entity> activeCollisions;
  private: std::optional<bool> currentAttached;
  private: std::atomic<int> stableState{-1};
  private: bool desiredAttached{false};
  private: bool transitionAttached{false};
  private: unsigned int settlingTicks{0u};
  private: Phase phase{Phase::kStable};
};
}  // namespace openeta::gazebo

GZ_ADD_PLUGIN(
    openeta::gazebo::AttachedCollisionFilter,
    gz::sim::System,
    gz::sim::ISystemConfigure,
    gz::sim::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
    openeta::gazebo::AttachedCollisionFilter,
    "openeta::gazebo::AttachedCollisionFilter")
