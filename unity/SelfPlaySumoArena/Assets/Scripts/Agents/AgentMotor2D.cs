using UnityEngine;

namespace SelfPlayArena.Agents
{
    [RequireComponent(typeof(Rigidbody2D))]
    public class AgentMotor2D : MonoBehaviour
    {
        [Header("Identity")]
        [SerializeField] private int agentId;

        [Header("Tuning Preset")]
        [SerializeField] private AgentMotorTuningPreset tuningPreset;
        [SerializeField] private bool syncWithPresetInEditor = true;

        [Header("Movement")]
        [SerializeField] private float moveSpeed = 4f;
        [SerializeField] private float maxSpeed = 6f;
        [SerializeField] private float idleDrag = 4f;

        [Header("Push")]
        [SerializeField] private float pushImpulse = 6f;
        [SerializeField] private float pushCooldown = 1.2f;
        [SerializeField] private float minInputMagnitude = 0.1f;
        [SerializeField] private float minPushSpeed = 0.2f;

        private Rigidbody2D rb;
        private AgentAction pendingAction;
        private float pushCooldownTimer;
        private bool usesExternalSimulationClock;

        public int AgentId => agentId;
        public Rigidbody2D Rigidbody => rb;
        public bool PushReady => pushCooldownTimer <= 0f;

        private void Awake()
        {
            ApplyTuningPreset();
            rb = GetComponent<Rigidbody2D>();
            pendingAction = AgentAction.Idle;
        }

        private void OnValidate()
        {
            if (!syncWithPresetInEditor)
            {
                return;
            }

            ApplyTuningPreset();
        }

        private void FixedUpdate()
        {
            if (usesExternalSimulationClock)
            {
                return;
            }

            TickMotor(Time.fixedDeltaTime);
        }

        public void SetPendingAction(AgentAction action)
        {
            pendingAction = action;
        }

        public void SetUsesExternalSimulationClock(bool value)
        {
            usesExternalSimulationClock = value;
        }

        public void TickMotor(float deltaTime)
        {
            if (pushCooldownTimer > 0f)
            {
                pushCooldownTimer -= deltaTime;
            }
        }

        public void ApplyPendingAction()
        {
            ApplyMove(pendingAction.move);
            TryApplyPush(pendingAction);
            pendingAction = AgentAction.Idle;
        }

        public void ResetAgent(Vector2 position)
        {
            transform.position = position;
            transform.rotation = Quaternion.identity;
            rb.velocity = Vector2.zero;
            rb.angularVelocity = 0f;
            pushCooldownTimer = 0f;
            pendingAction = AgentAction.Idle;
        }

        [ContextMenu("Apply Tuning Preset")]
        public void ApplyTuningPreset()
        {
            if (tuningPreset == null)
            {
                return;
            }

            moveSpeed = tuningPreset.moveSpeed;
            maxSpeed = tuningPreset.maxSpeed;
            idleDrag = tuningPreset.idleDrag;
            pushImpulse = tuningPreset.pushImpulse;
            pushCooldown = tuningPreset.pushCooldown;
            minInputMagnitude = tuningPreset.minInputMagnitude;
            minPushSpeed = tuningPreset.minPushSpeed;
        }

        private void ApplyMove(Vector2 moveInput)
        {
            Vector2 clamped = ClampInput(moveInput);
            rb.drag = clamped.sqrMagnitude > 0f ? 0f : idleDrag;
            rb.AddForce(clamped * moveSpeed, ForceMode2D.Force);

            if (rb.velocity.sqrMagnitude > maxSpeed * maxSpeed)
            {
                rb.velocity = rb.velocity.normalized * maxSpeed;
            }
        }

        private void TryApplyPush(AgentAction action)
        {
            if (!action.usePush || !PushReady)
            {
                return;
            }

            // Push direction depends only on current movement velocity.
            // This is less intuitive for a human player, but matches the
            // intended agent-facing action semantics.
            Vector2 pushDirection = GetVelocityDirection();
            if (pushDirection.sqrMagnitude <= 0f)
            {
                return;
            }

            rb.AddForce(pushDirection * pushImpulse, ForceMode2D.Impulse);
            pushCooldownTimer = pushCooldown;
        }

        private Vector2 GetVelocityDirection()
        {
            Vector2 velocity = rb.velocity;
            if (velocity.sqrMagnitude < minPushSpeed * minPushSpeed)
            {
                return Vector2.zero;
            }

            return velocity.normalized;
        }

        private Vector2 ClampInput(Vector2 input)
        {
            if (input.sqrMagnitude < minInputMagnitude * minInputMagnitude)
            {
                return Vector2.zero;
            }

            return input.sqrMagnitude > 1f ? input.normalized : input;
        }
    }
}
