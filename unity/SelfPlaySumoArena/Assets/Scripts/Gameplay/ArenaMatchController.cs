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
        [SerializeField] private RealtimeExternalAgentController2D realtimeAgent1Controller;
        [SerializeField] private Vector2 agent0Spawn = new Vector2(-1.5f, 0f);
        [SerializeField] private Vector2 agent1Spawn = new Vector2(1.5f, 0f);
        [SerializeField] private bool allowSideSwap = true;
        [SerializeField] private float spawnJitterRadius = 0.35f;
        [SerializeField] private float stepDuration = 0.05f;
        [SerializeField] private float episodeDuration = 400f;
        [SerializeField] private bool autoSimulateBridgeSteps = false;
        [SerializeField] private bool useManualPhysicsSimulation = true;
        [SerializeField] private bool logMatchResults = true;

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
        public bool UseManualPhysicsSimulation => useManualPhysicsSimulation;

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

        public void ResetMatch(int? resetSeed = null)
        {
            LogForcedResetIfNeeded();
            agent0.SetUsesExternalSimulationClock(useManualPhysicsSimulation);
            agent1.SetUsesExternalSimulationClock(useManualPhysicsSimulation);
            (Vector2 spawn0, Vector2 spawn1) = GetSpawnLayout(resetSeed);
            agent0.ResetAgent(GetWorldSpawnPosition(spawn0));
            agent1.ResetAgent(GetWorldSpawnPosition(spawn1));
            realtimeAgent1Controller?.ClearAction();
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

        public BridgeResponse ResetAndGetState(int resetSeed)
        {
            ResetMatch(resetSeed);
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

        public void PrepareBatchStep(AgentAction action0, AgentAction action1)
        {
            if (isDone)
            {
                return;
            }

            agent0.SetPendingAction(action0);
            agent1.SetPendingAction(action1);
        }

        public void ApplyPreparedBatchActions()
        {
            if (isDone)
            {
                return;
            }

            agent0.ApplyPendingAction();
            agent1.ApplyPendingAction();
        }

        public void TickBatchPrePhysics(float deltaTime)
        {
            if (isDone)
            {
                return;
            }

            agent0.TickMotor(deltaTime);
            agent1.TickMotor(deltaTime);
        }

        public void TickBatchPostPhysics(float deltaTime)
        {
            if (isDone)
            {
                return;
            }

            episodeTimer += deltaTime;
            CheckRingOut();

            if (!isDone && episodeTimer >= episodeDuration)
            {
                EndMatch(-1, "time_limit");
            }
        }

        public BridgeResponse SetRealtimeAgent1Action(AgentAction action)
        {
            if (realtimeAgent1Controller == null)
            {
                return BuildResponse("missing_realtime_agent1_controller");
            }

            realtimeAgent1Controller.SetLatestAction(action);
            return BuildResponse("set_agent1_action");
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
            LogMatchResult();
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
            Vector2 selfLocalPosition = ToArenaLocalPosition(self.transform.position);
            Vector2 opponentLocalPosition = ToArenaLocalPosition(opponent.transform.position);
            return new AgentObservation
            {
                selfPosition = selfLocalPosition,
                selfVelocity = self.Rigidbody.velocity,
                opponentPosition = opponentLocalPosition,
                opponentVelocity = opponent.Rigidbody.velocity,
                pushReady = self.PushReady
            };
        }

        private Vector2 GetWorldSpawnPosition(Vector2 localSpawn)
        {
            Transform arenaTransform = GetArenaTransform();
            return arenaTransform != null
                ? arenaTransform.TransformPoint(localSpawn)
                : localSpawn;
        }

        private Vector2 ToArenaLocalPosition(Vector2 worldPosition)
        {
            Transform arenaTransform = GetArenaTransform();
            return arenaTransform != null
                ? arenaTransform.InverseTransformPoint(worldPosition)
                : worldPosition;
        }

        private Transform GetArenaTransform()
        {
            return transform.parent != null ? transform.parent : null;
        }

        private void LogForcedResetIfNeeded()
        {
            if (!logMatchResults || isDone || episodeTimer <= 0f)
            {
                return;
            }

            Debug.Log(
                $"[{GetArenaLabel()}] reset_without_terminal elapsed={episodeTimer:F2}s " +
                $"reason=manual_or_external_reset");
        }

        private void LogMatchResult()
        {
            if (!logMatchResults)
            {
                return;
            }

            string outcome = winner switch
            {
                0 => "agent_0_win",
                1 => "agent_1_win",
                _ => "draw"
            };

            Debug.Log(
                $"[{GetArenaLabel()}] result={outcome} reason={terminalReason} " +
                $"elapsed={episodeTimer:F2}s reward0={reward0:F2} reward1={reward1:F2}");
        }

        private string GetArenaLabel()
        {
            Transform arenaTransform = GetArenaTransform();
            return arenaTransform != null ? arenaTransform.name : gameObject.name;
        }

        private (Vector2 spawn0, Vector2 spawn1) GetSpawnLayout(int? resetSeed)
        {
            Vector2 baseSpawn0 = agent0Spawn;
            Vector2 baseSpawn1 = agent1Spawn;

            if (!resetSeed.HasValue)
            {
                return (baseSpawn0, baseSpawn1);
            }

            System.Random rng = new System.Random(resetSeed.Value);
            bool swapSides = allowSideSwap && rng.NextDouble() < 0.5;
            Vector2 commonJitter = SampleDiskJitter(rng, spawnJitterRadius);

            if (swapSides)
            {
                return (baseSpawn1 + commonJitter, baseSpawn0 + commonJitter);
            }

            return (baseSpawn0 + commonJitter, baseSpawn1 + commonJitter);
        }

        private static Vector2 SampleDiskJitter(System.Random rng, float radius)
        {
            if (radius <= 0f)
            {
                return Vector2.zero;
            }

            double angle = rng.NextDouble() * Math.PI * 2.0;
            double distance = Math.Sqrt(rng.NextDouble()) * radius;
            return new Vector2(
                (float)(Math.Cos(angle) * distance),
                (float)(Math.Sin(angle) * distance));
        }
    }

    [Serializable]
    public class BridgeRequest
    {
        public string request_id = string.Empty;
        public string command = "get_state";
        public int[] arena_ids = Array.Empty<int>();
        public int[] arena_seeds = Array.Empty<int>();
        public BatchArenaActionPayload[] arenas = Array.Empty<BatchArenaActionPayload>();
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
