using System;
using SelfPlayArena.Agents;
using SelfPlayArena.Arena;
using UnityEngine;

namespace SelfPlayArena.Gameplay
{
    public class ArenaMatchController : MonoBehaviour
    {
        [SerializeField] private ArenaBoundary2D arenaBoundary;
        [SerializeField] private AgentMotor2D agent0;
        [SerializeField] private AgentMotor2D agent1;
        [SerializeField] private Vector2 agent0Spawn = new Vector2(-1.5f, 0f);
        [SerializeField] private Vector2 agent1Spawn = new Vector2(1.5f, 0f);
        [SerializeField] private float stepDuration = 0.1f;
        [SerializeField] private float episodeDuration = 30f;
        [SerializeField] private bool autoSimulateBridgeSteps = false;
        [SerializeField] private bool useManualPhysicsSimulation = true;

        private float stepTimer;
        private float episodeTimer;
        private bool isDone;
        private int winner = -1;
        private float reward0;
        private float reward1;
        private string terminalReason = "running";
        private bool stepRequested;
        private SimulationMode2D originalSimulationMode;

        public float StepDuration => stepDuration;
        public bool IsDone => isDone;
        public int Winner => winner;
        public bool AutoSimulateBridgeSteps => autoSimulateBridgeSteps;

        public event Action MatchStateChanged;

        private void Start()
        {
            originalSimulationMode = Physics2D.simulationMode;
            agent0.SetUsesExternalSimulationClock(useManualPhysicsSimulation);
            agent1.SetUsesExternalSimulationClock(useManualPhysicsSimulation);

            if (useManualPhysicsSimulation)
            {
                Physics2D.simulationMode = SimulationMode2D.Script;
            }

            ResetMatch();
        }

        private void OnDestroy()
        {
            if (useManualPhysicsSimulation)
            {
                Physics2D.simulationMode = originalSimulationMode;
            }
        }

        private void Update()
        {
            if (isDone)
            {
                return;
            }

            if (autoSimulateBridgeSteps && useManualPhysicsSimulation)
            {
                stepTimer += Time.deltaTime;
                while (stepTimer >= stepDuration)
                {
                    stepTimer -= stepDuration;
                    SimulateSingleStep(AgentAction.Idle, AgentAction.Idle);
                }
            }

            if (!useManualPhysicsSimulation)
            {
                episodeTimer += Time.deltaTime;
                if (episodeTimer >= episodeDuration)
                {
                    EndMatch(-1, "time_limit");
                }
            }
        }

        private void FixedUpdate()
        {
            if (useManualPhysicsSimulation)
            {
                return;
            }

            agent0.TickMotor(Time.fixedDeltaTime);
            agent1.TickMotor(Time.fixedDeltaTime);

            if (stepRequested)
            {
                agent0.ApplyPendingAction();
                agent1.ApplyPendingAction();
                stepRequested = false;
            }

            CheckRingOut();
        }

        public void ResetMatch()
        {
            agent0.SetUsesExternalSimulationClock(useManualPhysicsSimulation);
            agent1.SetUsesExternalSimulationClock(useManualPhysicsSimulation);
            agent0.ResetAgent(agent0Spawn);
            agent1.ResetAgent(agent1Spawn);
            reward0 = 0f;
            reward1 = 0f;
            winner = -1;
            isDone = false;
            terminalReason = "running";
            stepRequested = false;
            stepTimer = 0f;
            episodeTimer = 0f;
            MatchStateChanged?.Invoke();
        }

        public BridgeResponse ResetAndGetState()
        {
            ResetMatch();
            return BuildResponse("reset");
        }

        public BridgeResponse StepMatch(AgentAction action0, AgentAction action1)
        {
            if (!isDone)
            {
                if (useManualPhysicsSimulation)
                {
                    SimulateSingleStep(action0, action1);
                }
                else
                {
                    agent0.SetPendingAction(action0);
                    agent1.SetPendingAction(action1);
                    stepRequested = true;
                }
            }

            return BuildResponse("step");
        }

        public void AdvanceOneStep()
        {
            if (isDone)
            {
                return;
            }

            if (useManualPhysicsSimulation)
            {
                SimulateSingleStep(AgentAction.Idle, AgentAction.Idle);
            }
            else
            {
                stepRequested = true;
            }
        }

        public BridgeResponse GetCurrentState(string status = "ok")
        {
            return BuildResponse(status);
        }

        private void CheckRingOut()
        {
            if (isDone || arenaBoundary == null)
            {
                return;
            }

            bool agent0Out = arenaBoundary.IsOutOfBounds(agent0.transform.position);
            bool agent1Out = arenaBoundary.IsOutOfBounds(agent1.transform.position);

            if (agent0Out && agent1Out)
            {
                EndMatch(-1, "double_ring_out");
            }
            else if (agent0Out)
            {
                EndMatch(1, "agent_0_ring_out");
            }
            else if (agent1Out)
            {
                EndMatch(0, "agent_1_ring_out");
            }
        }

        private void SimulateSingleStep(AgentAction action0, AgentAction action1)
        {
            agent0.SetPendingAction(action0);
            agent1.SetPendingAction(action1);
            agent0.ApplyPendingAction();
            agent1.ApplyPendingAction();

            int simulationTicks = Mathf.Max(1, Mathf.CeilToInt(stepDuration / Time.fixedDeltaTime));
            float simulationDelta = stepDuration / simulationTicks;

            for (int i = 0; i < simulationTicks; i++)
            {
                agent0.TickMotor(simulationDelta);
                agent1.TickMotor(simulationDelta);
                Physics2D.Simulate(simulationDelta);
                episodeTimer += simulationDelta;
                CheckRingOut();

                if (isDone)
                {
                    return;
                }
            }

            if (episodeTimer >= episodeDuration)
            {
                EndMatch(-1, "time_limit");
            }
        }

        private void EndMatch(int matchWinner, string reason)
        {
            if (isDone)
            {
                return;
            }

            isDone = true;
            winner = matchWinner;
            terminalReason = string.IsNullOrWhiteSpace(reason) ? "unknown" : reason;
            reward0 = matchWinner == 0 ? 1f : matchWinner == 1 ? -1f : 0f;
            reward1 = matchWinner == 1 ? 1f : matchWinner == 0 ? -1f : 0f;
            MatchStateChanged?.Invoke();
        }

        private BridgeResponse BuildResponse(string status)
        {
            return new BridgeResponse
            {
                status = status,
                done = isDone,
                winner = winner,
                reward0 = reward0,
                reward1 = reward1,
                terminalReason = terminalReason,
                agent0 = BuildObservation(agent0, agent1),
                agent1 = BuildObservation(agent1, agent0)
            };
        }

        private AgentObservation BuildObservation(AgentMotor2D self, AgentMotor2D opponent)
        {
            return new AgentObservation
            {
                selfPosition = self.transform.position,
                selfVelocity = self.Rigidbody.velocity,
                opponentPosition = opponent.transform.position,
                opponentVelocity = opponent.Rigidbody.velocity,
                pushReady = self.PushReady
            };
        }
    }

    [Serializable]
    public class BridgeRequest
    {
        public string request_id = string.Empty;
        public string command = "get_state";
        public AgentActionPayload agent0 = new AgentActionPayload();
        public AgentActionPayload agent1 = new AgentActionPayload();
    }

    [Serializable]
    public class AgentActionPayload
    {
        public float[] move = new float[2];
        public float[] push = new float[2];
        public bool use_push;

        public AgentAction ToAgentAction()
        {
            return new AgentAction
            {
                move = ToVector2(move),
                push = ToVector2(push),
                usePush = use_push
            };
        }

        private static Vector2 ToVector2(float[] values)
        {
            if (values == null || values.Length < 2)
            {
                return Vector2.zero;
            }

            return new Vector2(values[0], values[1]);
        }
    }

    [Serializable]
    public class BridgeResponse
    {
        public string request_id;
        public string status;
        public bool done;
        public int winner;
        public float reward0;
        public float reward1;
        public string terminalReason;
        public AgentObservation agent0;
        public AgentObservation agent1;
    }
}
