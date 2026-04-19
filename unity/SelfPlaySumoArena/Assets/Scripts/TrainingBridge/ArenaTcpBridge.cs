using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using SelfPlayArena.Gameplay;
using UnityEngine;

namespace SelfPlayArena.TrainingBridge
{
    public class ArenaTcpBridge : MonoBehaviour
    {
        [SerializeField] private ArenaMatchController matchController;
        [SerializeField] private ArenaBatchManager batchManager;
        [SerializeField] private int port = 5055;
        [SerializeField] private bool startOnAwake = true;
        [SerializeField] private bool runInBackground = true;
        [SerializeField] private bool verboseLogging;

        private TcpListener listener;
        private TcpClient activeClient;
        private Socket activeSocket;
        private StringBuilder receiveBuffer = new StringBuilder();
        private bool previousRunInBackground;

        private void Awake()
        {
            previousRunInBackground = Application.runInBackground;
            if (runInBackground)
            {
                Application.runInBackground = true;
            }

            if (startOnAwake)
            {
                StartServer();
            }
        }

        private void Update()
        {
            if (listener == null)
            {
                return;
            }

            AcceptPendingClient();
            ServiceActiveClient();
        }

        private void OnDestroy()
        {
            StopServer();
            Application.runInBackground = previousRunInBackground;
        }

        [ContextMenu("Start Server")]
        public void StartServer()
        {
            if (listener != null)
            {
                return;
            }

            listener = new TcpListener(IPAddress.Loopback, port);
            listener.Start();

            if (verboseLogging)
            {
                Debug.Log($"ArenaTcpBridge started on port {port}");
            }
        }

        [ContextMenu("Stop Server")]
        public void StopServer()
        {
            CloseActiveClient();

            if (listener == null)
            {
                return;
            }

            listener.Stop();
            listener = null;

            if (verboseLogging)
            {
                Debug.Log("ArenaTcpBridge stopped");
            }
        }

        private void AcceptPendingClient()
        {
            if (activeSocket != null || listener == null || !listener.Pending())
            {
                return;
            }

            try
            {
                activeClient = listener.AcceptTcpClient();
                activeSocket = activeClient.Client;
                activeSocket.NoDelay = true;
                activeSocket.ReceiveTimeout = 2000;
                activeSocket.SendTimeout = 2000;
                receiveBuffer.Clear();

                if (verboseLogging)
                {
                    Debug.Log("ArenaTcpBridge accepted persistent client connection.");
                }
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"ArenaTcpBridge accept error: {exception}");
                CloseActiveClient();
            }
        }

        private void ServiceActiveClient()
        {
            if (activeSocket == null)
            {
                return;
            }

            try
            {
                if (IsRemoteClosed(activeSocket))
                {
                    if (verboseLogging)
                    {
                        Debug.Log("ArenaTcpBridge client disconnected.");
                    }

                    CloseActiveClient();
                    return;
                }

                ReadAvailableSocketData(activeSocket, receiveBuffer);

                while (TryPopLine(receiveBuffer, out string requestJson))
                {
                    if (string.IsNullOrWhiteSpace(requestJson))
                    {
                        continue;
                    }

                    if (verboseLogging)
                    {
                        Debug.Log($"ArenaTcpBridge received: {requestJson}");
                        Debug.Log("ArenaTcpBridge handling request on main thread.");
                    }

                    string responseJson = HandleRequest(requestJson);

                    if (verboseLogging)
                    {
                        Debug.Log($"ArenaTcpBridge response prepared: {responseJson}");
                    }

                    int bytesSent = WriteJsonLine(activeSocket, responseJson);

                    if (verboseLogging)
                    {
                        Debug.Log($"ArenaTcpBridge wrote {bytesSent} bytes.");
                        Debug.Log($"ArenaTcpBridge wrote response: {responseJson}");
                    }
                }
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"ArenaTcpBridge client handling error: {exception}");
                CloseActiveClient();
            }
        }

        private static bool IsRemoteClosed(Socket socket)
        {
            return socket.Poll(0, SelectMode.SelectRead) && socket.Available == 0;
        }

        private static void ReadAvailableSocketData(Socket socket, StringBuilder buffer)
        {
            byte[] receiveBytes = new byte[4096];

            while (socket.Available > 0)
            {
                int received = socket.Receive(receiveBytes, 0, receiveBytes.Length, SocketFlags.None);
                if (received <= 0)
                {
                    break;
                }

                buffer.Append(Encoding.UTF8.GetString(receiveBytes, 0, received));
            }
        }

        private static bool TryPopLine(StringBuilder buffer, out string line)
        {
            string current = buffer.ToString();
            int newlineIndex = current.IndexOf('\n');
            if (newlineIndex < 0)
            {
                line = null;
                return false;
            }

            line = current.Substring(0, newlineIndex).TrimEnd('\r');
            buffer.Remove(0, newlineIndex + 1);
            return true;
        }

        private void CloseActiveClient()
        {
            receiveBuffer.Clear();

            try
            {
                activeSocket?.Close();
            }
            catch
            {
            }

            try
            {
                activeClient?.Close();
            }
            catch
            {
            }

            activeSocket = null;
            activeClient = null;
        }

        private string HandleRequest(string requestJson)
        {
            BridgeRequest request;
            try
            {
                request = JsonUtility.FromJson<BridgeRequest>(requestJson);
            }
            catch (Exception)
            {
                return JsonUtility.ToJson(new BridgeResponse { status = "invalid_json", winner = -1 });
            }

            if (request == null)
            {
                return JsonUtility.ToJson(new BridgeResponse { status = "invalid_request", winner = -1 });
            }

            try
            {
                return request.command switch
                {
                    "reset_batch" => BatchResponseJson(() => batchManager.ResetAllAndGetState(request.arena_seeds), "missing_batch_manager"),
                    "reset_arenas" => BatchResponseJson(() => batchManager.ResetArenas(request.arena_ids, request.arena_seeds), "missing_batch_manager"),
                    "step_batch" => BatchResponseJson(() => batchManager.StepBatch(request.arenas), "missing_batch_manager"),
                    "get_batch_state" => BatchResponseJson(() => batchManager.GetBatchState("get_batch_state"), "missing_batch_manager"),
                    "reset" => SingleResponseJson(() => matchController.ResetAndGetState(), "missing_match_controller"),
                    "step" => SingleResponseJson(
                        () => matchController.StepMatch(
                            request.agent0?.ToAgentAction() ?? new AgentActionPayload().ToAgentAction(),
                            request.agent1?.ToAgentAction() ?? new AgentActionPayload().ToAgentAction()),
                        "missing_match_controller"),
                    "get_state" => SingleResponseJson(() => matchController.GetCurrentState(), "missing_match_controller"),
                    "set_agent1_action" => SingleResponseJson(
                        () => matchController.SetRealtimeAgent1Action(
                            request.agent1?.ToAgentAction() ?? new AgentActionPayload().ToAgentAction()),
                        "missing_match_controller"),
                    _ => JsonUtility.ToJson(new BridgeResponse { status = "unknown_command", winner = -1 })
                };
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"ArenaTcpBridge request handling error: {exception}");
                return JsonUtility.ToJson(new BridgeResponse
                {
                    status = $"handler_exception:{exception.GetType().Name}",
                    winner = -1
                });
            }
        }

        private string SingleResponseJson(Func<BridgeResponse> buildResponse, string missingStatus)
        {
            if (matchController == null)
            {
                return JsonUtility.ToJson(new BridgeResponse { status = missingStatus, winner = -1 });
            }

            return JsonUtility.ToJson(buildResponse());
        }

        private string BatchResponseJson(Func<BatchBridgeResponse> buildResponse, string missingStatus)
        {
            if (batchManager == null)
            {
                return JsonUtility.ToJson(new BatchBridgeResponse { status = missingStatus });
            }

            return JsonUtility.ToJson(buildResponse());
        }

        private static int WriteJsonLine(Socket socket, string json)
        {
            string payload = (json ?? "{\"status\":\"empty_response\",\"winner\":-1}") + "\n";
            byte[] bytes = Encoding.UTF8.GetBytes(payload);
            int sent = 0;

            while (sent < bytes.Length)
            {
                int justSent = socket.Send(bytes, sent, bytes.Length - sent, SocketFlags.None);
                if (justSent <= 0)
                {
                    throw new SocketException((int)SocketError.ConnectionReset);
                }

                sent += justSent;
            }

            return sent;
        }
    }
}
