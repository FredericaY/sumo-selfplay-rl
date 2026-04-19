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
            Vector2 worldCenter = transform.TransformPoint(center);
            return Vector2.Distance(worldCenter, worldPosition) > radius;
        }

        private void OnDrawGizmos()
        {
            Gizmos.color = Color.yellow;
            Matrix4x4 previousMatrix = Gizmos.matrix;
            Gizmos.matrix = transform.localToWorldMatrix;
            Gizmos.DrawWireSphere(new Vector3(center.x, center.y, 0f), radius);
            Gizmos.matrix = previousMatrix;
        }
    }
}
