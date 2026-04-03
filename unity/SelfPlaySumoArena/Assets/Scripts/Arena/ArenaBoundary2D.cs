using UnityEngine;

namespace SelfPlayArena.Arena
{
    public class ArenaBoundary2D : MonoBehaviour
    {
        [SerializeField] private float radius = 5f;
        [SerializeField] private Vector2 center = Vector2.zero;

        public float Radius => radius;
        public Vector2 Center => center;

        public bool IsOutOfBounds(Vector2 worldPosition)
        {
            return Vector2.Distance(center, worldPosition) > radius;
        }

        private void OnDrawGizmos()
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(new Vector3(center.x, center.y, 0f), radius);
        }
    }
}
