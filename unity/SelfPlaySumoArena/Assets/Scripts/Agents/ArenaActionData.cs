using System;
using UnityEngine;

namespace SelfPlayArena.Agents
{
    [Serializable]
    public struct AgentAction
    {
        public Vector2 move;
        public Vector2 push;
        public bool usePush;

        public static AgentAction Idle => new AgentAction
        {
            move = Vector2.zero,
            push = Vector2.zero,
            usePush = false
        };
    }

    [Serializable]
    public struct AgentObservation
    {
        public Vector2 selfPosition;
        public Vector2 selfVelocity;
        public Vector2 opponentPosition;
        public Vector2 opponentVelocity;
        public bool pushReady;
    }
}
