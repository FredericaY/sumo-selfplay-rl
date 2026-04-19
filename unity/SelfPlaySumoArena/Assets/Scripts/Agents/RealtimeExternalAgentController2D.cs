using UnityEngine;

namespace SelfPlayArena.Agents
{
    public class RealtimeExternalAgentController2D : MonoBehaviour
    {
        [SerializeField] private AgentMotor2D motor;
        [SerializeField] private bool applyLatestActionInFixedUpdate = true;

        private AgentAction latestAction = AgentAction.Idle;

        public AgentMotor2D Motor => motor;

        private void Reset()
        {
            motor = GetComponent<AgentMotor2D>();
        }

        public void SetLatestAction(AgentAction action)
        {
            latestAction = action;
        }

        public void ClearAction()
        {
            latestAction = AgentAction.Idle;
        }

        private void FixedUpdate()
        {
            if (!applyLatestActionInFixedUpdate || motor == null)
            {
                return;
            }

            motor.SetPendingAction(latestAction);
            motor.ApplyPendingAction();
        }
    }
}
