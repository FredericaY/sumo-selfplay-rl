using UnityEngine;

namespace SelfPlayArena.Agents
{
    [CreateAssetMenu(
        fileName = "AgentMotorTuningPreset",
        menuName = "SelfPlayArena/Agent Motor Tuning Preset")]
    public class AgentMotorTuningPreset : ScriptableObject
    {
        [Header("Movement")]
        public float moveSpeed = 4f;
        public float maxSpeed = 6f;
        public float idleDrag = 4f;

        [Header("Push")]
        public float pushImpulse = 6f;
        public float pushCooldown = 1.2f;
        public float minInputMagnitude = 0.1f;
        public float minPushSpeed = 0.2f;
    }
}
