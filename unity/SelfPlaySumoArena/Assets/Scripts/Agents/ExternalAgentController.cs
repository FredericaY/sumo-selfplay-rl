using UnityEngine;

namespace SelfPlayArena.Agents
{
    public class ExternalAgentController : MonoBehaviour
    {
        [SerializeField] private AgentMotor2D motor;

        public AgentMotor2D Motor => motor;

        private void Reset()
        {
            motor = GetComponent<AgentMotor2D>();
        }

        public void SetAction(AgentAction action)
        {
            if (motor == null)
            {
                return;
            }

            motor.SetPendingAction(action);
        }
    }
}
