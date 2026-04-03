using UnityEngine;

namespace SelfPlayArena.Agents
{
    public class HumanInputAgent2D : MonoBehaviour
    {
        [SerializeField] private AgentMotor2D motor;
        [SerializeField] private KeyCode upKey = KeyCode.W;
        [SerializeField] private KeyCode downKey = KeyCode.S;
        [SerializeField] private KeyCode leftKey = KeyCode.A;
        [SerializeField] private KeyCode rightKey = KeyCode.D;
        [SerializeField] private KeyCode pushKey = KeyCode.Space;

        private Vector2 currentMoveInput;
        private bool pushQueued;

        private void Reset()
        {
            motor = GetComponent<AgentMotor2D>();
        }

        private void Update()
        {
            if (motor == null)
            {
                return;
            }

            currentMoveInput = ReadMoveInput();

            // Latch the push request until the next FixedUpdate so we do not
            // lose the trigger when multiple Update calls happen between
            // physics ticks.
            if (Input.GetKeyDown(pushKey))
            {
                pushQueued = true;
            }
        }

        private void FixedUpdate()
        {
            if (motor == null)
            {
                return;
            }

            AgentAction action = new AgentAction
            {
                move = currentMoveInput,
                push = Vector2.zero,
                usePush = pushQueued
            };

            motor.SetPendingAction(action);
            motor.ApplyPendingAction();
            pushQueued = false;
        }
        
        private Vector2 ReadMoveInput()
        {
            float x = 0f;
            float y = 0f;

            if (Input.GetKey(leftKey))
            {
                x -= 1f;
            }

            if (Input.GetKey(rightKey))
            {
                x += 1f;
            }

            if (Input.GetKey(downKey))
            {
                y -= 1f;
            }

            if (Input.GetKey(upKey))
            {
                y += 1f;
            }

            Vector2 input = new Vector2(x, y);
            return input.sqrMagnitude > 1f ? input.normalized : input;
        }
    }
}
