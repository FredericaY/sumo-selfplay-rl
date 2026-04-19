using UnityEngine;

namespace SelfPlayArena.Agents
{
    [CreateAssetMenu(
        fileName = "AgentMotorTuningPreset",
        menuName = "SelfPlayArena/Agent Motor Tuning Preset")]
    public class AgentMotorTuningPreset : ScriptableObject
    {
        [Header("Movement")]
        public float moveSpeed = 1.7f;
        public float maxSpeed = 2.8f;
        public float moveDrag = 0.9f;
        public float idleDrag = 2.6f;

        [Header("Push")]
        public float pushImpulse = 3.4f;
        public float pushCooldown = 2.4f;
        public float pushRecoveryDrag = 3.2f;
        public float pushRecoveryDuration = 0.45f;
        public float maxPushSpeed = 5f;
        public float minInputMagnitude = 0.1f;
        public float minPushSpeed = 0.12f;
    }
}
