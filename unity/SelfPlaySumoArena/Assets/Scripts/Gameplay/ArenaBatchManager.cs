using System;
using System.Collections.Generic;
using SelfPlayArena.Agents;
using UnityEngine;

namespace SelfPlayArena.Gameplay
{
    public class ArenaBatchManager : MonoBehaviour
    {
        [SerializeField] private ArenaMatchController[] arenaControllers = Array.Empty<ArenaMatchController>();

        private readonly Dictionary<int, ArenaMatchController> controllerByArenaId = new Dictionary<int, ArenaMatchController>();
        private SimulationMode2D previousSimulationMode;

        private void Awake()
        {
            previousSimulationMode = Physics2D.simulationMode;
            Physics2D.simulationMode = SimulationMode2D.Script;
            RebuildArenaLookup();
        }

        private void OnValidate()
        {
            RebuildArenaLookup();
        }

        private void OnDestroy()
        {
            Physics2D.simulationMode = previousSimulationMode;
        }

        public BatchBridgeResponse ResetAllAndGetState(int[] arenaSeeds = null)
        {
            BatchArenaState[] arenas = new BatchArenaState[arenaControllers.Length];
            for (int i = 0; i < arenaControllers.Length; i++)
            {
                int? resetSeed = TryGetSeed(arenaSeeds, i);
                arenas[i] = ToBatchArenaState(
                    i,
                    resetSeed.HasValue
                        ? arenaControllers[i].ResetAndGetState(resetSeed.Value)
                        : arenaControllers[i].ResetAndGetState());
            }

            return new BatchBridgeResponse
            {
                status = "reset_batch",
                arenas = arenas
            };
        }

        public BatchBridgeResponse ResetArenas(int[] arenaIds, int[] arenaSeeds = null)
        {
            if (arenaIds == null || arenaIds.Length == 0)
            {
                return GetBatchState("reset_arenas");
            }

            List<BatchArenaState> states = new List<BatchArenaState>(arenaIds.Length);
            for (int requestIdx = 0; requestIdx < arenaIds.Length; requestIdx++)
            {
                int arenaId = arenaIds[requestIdx];
                if (!controllerByArenaId.TryGetValue(arenaId, out ArenaMatchController controller))
                {
                    continue;
                }

                int? resetSeed = TryGetSeed(arenaSeeds, requestIdx);
                states.Add(
                    ToBatchArenaState(
                        arenaId,
                        resetSeed.HasValue
                            ? controller.ResetAndGetState(resetSeed.Value)
                            : controller.ResetAndGetState()));
            }

            return new BatchBridgeResponse
            {
                status = "reset_arenas",
                arenas = states.ToArray()
            };
        }

        public BatchBridgeResponse StepBatch(BatchArenaActionPayload[] arenas)
        {
            if (arenas == null || arenas.Length == 0)
            {
                return GetBatchState("step_batch");
            }

            float batchStepDuration = ResolveBatchStepDuration();

            foreach (BatchArenaActionPayload arenaPayload in arenas)
            {
                if (arenaPayload == null)
                {
                    continue;
                }

                if (!controllerByArenaId.TryGetValue(arenaPayload.arena_id, out ArenaMatchController controller))
                {
                    continue;
                }

                controller.PrepareBatchStep(
                    arenaPayload.agent0?.ToAgentAction() ?? new AgentActionPayload().ToAgentAction(),
                    arenaPayload.agent1?.ToAgentAction() ?? new AgentActionPayload().ToAgentAction());
            }

            SimulateBatchStep(batchStepDuration);

            return new BatchBridgeResponse
            {
                status = "step_batch",
                arenas = BuildAllArenaStates()
            };
        }

        public BatchBridgeResponse GetBatchState(string status = "ok")
        {
            BatchArenaState[] arenas = new BatchArenaState[arenaControllers.Length];
            for (int i = 0; i < arenaControllers.Length; i++)
            {
                arenas[i] = ToBatchArenaState(i, arenaControllers[i].GetCurrentState(status));
            }

            return new BatchBridgeResponse
            {
                status = status,
                arenas = arenas
            };
        }

        private void RebuildArenaLookup()
        {
            controllerByArenaId.Clear();
            if (arenaControllers == null)
            {
                return;
            }

            for (int i = 0; i < arenaControllers.Length; i++)
            {
                if (arenaControllers[i] == null)
                {
                    continue;
                }

                if (!arenaControllers[i].UseManualPhysicsSimulation)
                {
                    Debug.LogWarning(
                        $"ArenaBatchManager detected {arenaControllers[i].name} with Use Manual Physics Simulation disabled. " +
                        "Batch stepping expects this to be enabled on every arena.");
                }

                controllerByArenaId[i] = arenaControllers[i];
            }
        }

        private float ResolveBatchStepDuration()
        {
            if (arenaControllers == null || arenaControllers.Length == 0 || arenaControllers[0] == null)
            {
                return Time.fixedDeltaTime;
            }

            return Mathf.Max(Time.fixedDeltaTime, arenaControllers[0].StepDuration);
        }

        private void SimulateBatchStep(float batchStepDuration)
        {
            if (arenaControllers == null || arenaControllers.Length == 0)
            {
                return;
            }

            for (int i = 0; i < arenaControllers.Length; i++)
            {
                ArenaMatchController controller = arenaControllers[i];
                if (controller == null)
                {
                    continue;
                }

                controller.ApplyPreparedBatchActions();
            }

            int simulationTicks = Mathf.Max(1, Mathf.CeilToInt(batchStepDuration / Time.fixedDeltaTime));
            float simulationDelta = batchStepDuration / simulationTicks;

            for (int tick = 0; tick < simulationTicks; tick++)
            {
                for (int i = 0; i < arenaControllers.Length; i++)
                {
                    ArenaMatchController controller = arenaControllers[i];
                    if (controller == null)
                    {
                        continue;
                    }

                    controller.TickBatchPrePhysics(simulationDelta);
                }

                Physics2D.Simulate(simulationDelta);

                for (int i = 0; i < arenaControllers.Length; i++)
                {
                    ArenaMatchController controller = arenaControllers[i];
                    if (controller == null)
                    {
                        continue;
                    }

                    controller.TickBatchPostPhysics(simulationDelta);
                }
            }
        }

        private BatchArenaState[] BuildAllArenaStates()
        {
            BatchArenaState[] arenas = new BatchArenaState[arenaControllers.Length];
            for (int i = 0; i < arenaControllers.Length; i++)
            {
                arenas[i] = ToBatchArenaState(i, arenaControllers[i].GetCurrentState("step_batch"));
            }

            return arenas;
        }

        private static BatchArenaState ToBatchArenaState(int arenaId, BridgeResponse response)
        {
            return new BatchArenaState
            {
                arena_id = arenaId,
                done = response.done,
                winner = response.winner,
                reward0 = response.reward0,
                reward1 = response.reward1,
                terminalReason = response.terminalReason,
                agent0 = response.agent0,
                agent1 = response.agent1
            };
        }

        private static int? TryGetSeed(int[] arenaSeeds, int index)
        {
            if (arenaSeeds == null || index < 0 || index >= arenaSeeds.Length)
            {
                return null;
            }

            return arenaSeeds[index];
        }
    }

    [Serializable]
    public class BatchArenaActionPayload
    {
        public int arena_id;
        public AgentActionPayload agent0 = new AgentActionPayload();
        public AgentActionPayload agent1 = new AgentActionPayload();
    }

    [Serializable]
    public class BatchArenaState
    {
        public int arena_id;
        public bool done;
        public int winner;
        public float reward0;
        public float reward1;
        public string terminalReason;
        public AgentObservation agent0;
        public AgentObservation agent1;
    }

    [Serializable]
    public class BatchBridgeResponse
    {
        public string status;
        public BatchArenaState[] arenas = Array.Empty<BatchArenaState>();
    }
}
