using System;
using System.IO;
using System.Linq;
using SelfPlayArena.Gameplay;
using UnityEngine;

namespace SelfPlayArena.TrainingBridge
{
    public class ArenaFileBridge : MonoBehaviour
    {
        [SerializeField] private ArenaMatchController matchController;
        [SerializeField] private string bridgeFolderName = "bridge_io";
        [SerializeField] private bool createFolderOnAwake = true;
        [SerializeField] private bool verboseLogging;

        private string bridgeDirectory;

        private void Awake()
        {
            bridgeDirectory = Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", "..", bridgeFolderName));

            if (createFolderOnAwake)
            {
                Directory.CreateDirectory(bridgeDirectory);
            }

            if (verboseLogging)
            {
                Debug.Log($"ArenaFileBridge watching: {bridgeDirectory}");
            }
        }

        private void Update()
        {
            string[] requestFiles = Directory.GetFiles(bridgeDirectory, "request_*.json")
                .OrderBy(path => path, StringComparer.Ordinal)
                .ToArray();

            if (requestFiles.Length == 0)
            {
                return;
            }

            foreach (string requestFile in requestFiles)
            {
                HandleRequestFile(requestFile);
            }
        }

        private void HandleRequestFile(string requestPath)
        {
            try
            {
                string requestJson = File.ReadAllText(requestPath);
                if (string.IsNullOrWhiteSpace(requestJson))
                {
                    File.Delete(requestPath);
                    return;
                }

                if (verboseLogging)
                {
                    Debug.Log($"ArenaFileBridge received: {requestJson}");
                }

                BridgeRequest request = JsonUtility.FromJson<BridgeRequest>(requestJson);
                BridgeResponse response = HandleRequest(request);
                response.request_id = request?.request_id ?? string.Empty;

                string requestId = string.IsNullOrWhiteSpace(response.request_id)
                    ? Guid.NewGuid().ToString("N")
                    : response.request_id;
                string responsePath = Path.Combine(bridgeDirectory, $"response_{requestId}.json");
                string tempPath = responsePath + ".tmp";
                string responseJson = JsonUtility.ToJson(response);

                File.WriteAllText(tempPath, responseJson);
                if (File.Exists(responsePath))
                {
                    File.Delete(responsePath);
                }

                File.Move(tempPath, responsePath);
                File.Delete(requestPath);

                if (verboseLogging)
                {
                    Debug.Log($"ArenaFileBridge wrote: {responseJson}");
                }
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"ArenaFileBridge error: {exception}");
            }
        }

        private BridgeResponse HandleRequest(BridgeRequest request)
        {
            if (matchController == null)
            {
                return new BridgeResponse
                {
                    status = "missing_match_controller",
                    winner = -1
                };
            }

            if (request == null)
            {
                return new BridgeResponse
                {
                    status = "invalid_request",
                    winner = -1
                };
            }

            return request.command switch
            {
                "reset" => WithRequestId(matchController.ResetAndGetState(), request.request_id),
                "step" => WithRequestId(
                    matchController.StepMatch(
                        request.agent0?.ToAgentAction() ?? new AgentActionPayload().ToAgentAction(),
                        request.agent1?.ToAgentAction() ?? new AgentActionPayload().ToAgentAction()),
                    request.request_id),
                "get_state" => WithRequestId(matchController.GetCurrentState(), request.request_id),
                _ => WithRequestId(matchController.GetCurrentState("unknown_command"), request.request_id)
            };
        }

        private static BridgeResponse WithRequestId(BridgeResponse response, string requestId)
        {
            response.request_id = requestId;
            return response;
        }
    }
}
