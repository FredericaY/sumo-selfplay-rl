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
        [SerializeField] private int port = 5055;
        [SerializeField] private bool startOnAwake = true;
        [SerializeField] private bool runInBackground = true;
        [SerializeField] private bool verboseLogging;

        private TcpListener listener;
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

            while (listener.Pending())
            {
                HandleSingleClient();
            }
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

        private void HandleSingleClient()
        {
            try
            {
                using TcpClient client = listener.AcceptTcpClient();
                Socket socket = client.Client;
                socket.NoDelay = true;
                socket.ReceiveTimeout = 2000;
                socket.SendTimeout = 2000;

                string requestJson = ReadLineFromSocket(socket);
                if (string.IsNullOrWhiteSpace(requestJson))
                {
                    return;
                }

                if (verboseLogging)
                {
                    Debug.Log($"ArenaTcpBridge received: {requestJson}");
                    Debug.Log("ArenaTcpBridge handling request on main thread.");
                }

                BridgeResponse response = HandleRequest(requestJson);
                string responseJson = JsonUtility.ToJson(response);

                if (verboseLogging)
                {
                    Debug.Log($"ArenaTcpBridge response prepared: {responseJson}");
                }

                int bytesSent = WriteJsonLine(socket, responseJson);

                if (verboseLogging)
                {
                    Debug.Log($"ArenaTcpBridge wrote {bytesSent} bytes.");
                    Debug.Log($"ArenaTcpBridge wrote response: {responseJson}");
                }
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"ArenaTcpBridge client handling error: {exception}");
            }
        }

        private BridgeResponse HandleRequest(string requestJson)
        {
            if (matchController == null)
            {
                return new BridgeResponse { status = "missing_match_controller", winner = -1 };
            }

            BridgeRequest request;
            try
            {
                request = JsonUtility.FromJson<BridgeRequest>(requestJson);
            }
            catch (Exception)
            {
                return new BridgeResponse { status = "invalid_json", winner = -1 };
            }

            if (request == null)
            {
                return new BridgeResponse { status = "invalid_request", winner = -1 };
            }

            try
            {
                return request.command switch
                {
                    "reset" => matchController.ResetAndGetState(),
                    "step" => matchController.StepMatch(
                        request.agent0?.ToAgentAction() ?? new AgentActionPayload().ToAgentAction(),
                        request.agent1?.ToAgentAction() ?? new AgentActionPayload().ToAgentAction()),
                    "get_state" => matchController.GetCurrentState(),
                    _ => matchController.GetCurrentState("unknown_command")
                };
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"ArenaTcpBridge request handling error: {exception}");
                return new BridgeResponse
                {
                    status = $"handler_exception:{exception.GetType().Name}",
                    winner = -1
                };
            }
        }

        private static string ReadLineFromSocket(Socket socket)
        {
            byte[] buffer = new byte[4096];
            StringBuilder builder = new StringBuilder();

            while (true)
            {
                int received = socket.Receive(buffer, 0, buffer.Length, SocketFlags.None);
                if (received <= 0)
                {
                    break;
                }

                string chunk = Encoding.UTF8.GetString(buffer, 0, received);
                builder.Append(chunk);

                int newlineIndex = builder.ToString().IndexOf('\n');
                if (newlineIndex >= 0)
                {
                    return builder.ToString(0, newlineIndex).TrimEnd('\r');
                }
            }

            return builder.ToString().Trim();
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
