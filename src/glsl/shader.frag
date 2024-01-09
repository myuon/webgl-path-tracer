#version 300 es

precision highp float;

uniform sampler2D u_texture;
uniform int iterations;
uniform vec2 resolution;
uniform vec3 camera_position;
uniform vec3 camera_direction;
uniform vec3 camera_up;
uniform float screen_dist;
uniform int spp;
uniform int render_type;
uniform sampler2D triangles_texture;
uniform sampler2D material_texture;
uniform sampler2D bvh_tree_texture;

const int RenderTypeRender = 0;
const int RenderTypeColor = 1;
const int RenderTypeNormal = 2;

in vec2 v_texcoord;
out vec4 outColor;

const float PI = 3.14159265;
const float kEPS = 1e-6;

highp float rand(vec2 co){
    highp float a = 12.9898;
    highp float b = 78.233;
    highp float c = 43758.5453;
    highp float dt= dot(co.xy ,vec2(a,b));
    highp float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
}

vec3 rand3(vec3 p){
	vec3 q = vec3(
		dot(p, vec3(127.1, 311.7, 74.7)),
		dot(p, vec3(269.5, 183.3, 246.1)),
		dot(p, vec3(113.5, 271.9, 124.6))
		);

	return fract(sin(q) * 43758.5453123);
}

vec3 randOnHemisphere(vec3 n, float seed){
    vec3 w = n;
    vec3 u = normalize(cross(vec3(1.0, 0.0, 0.0), w));
    if (abs(w.x) > kEPS) {
        u = normalize(cross(vec3(0.0, 1.0, 0.0), w));
    }

    vec3 v = cross(w, u);
    float r1 = rand(vec2(seed, 0.0) + n.xy);
    float r2 = rand(vec2(seed, 1.0) + n.yz);

    float phy = 2.0 * PI * r1;
    float cos_theta = sqrt(r2);

    return normalize(u * cos(phy) * cos_theta + v * sin(phy) * cos_theta + w * sqrt(1.0 - r2));
}

struct HitRecord {
    bool hit;
    vec3 normal;
    vec3 point;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Triangle {
    vec3 vertex;
    vec3 edge1;
    vec3 edge2;
    vec3 normal0;
    vec3 normal1;
    vec3 normal2;
    int material_id;
    bool smooth_normal;
};

float det(vec3 a, vec3 b, vec3 c) {
    return a.x * b.y * c.z + a.y * b.z * c.x + a.z * b.x * c.y
        - a.z * b.y * c.x - a.y * b.x * c.z - a.x * b.z * c.y;
}

HitRecord Triangle_intersect(Triangle self, Ray ray) {
    float d = det(self.edge1, self.edge2, -ray.direction);
    if (abs(d) < kEPS) {
        return HitRecord(false, vec3(0.0), vec3(0.0));
    }

    vec3 ov = ray.origin - self.vertex;

    float u = det(ov, self.edge2, -ray.direction) / d;
    float v = det(self.edge1, ov, -ray.direction) / d;
    float t = det(self.edge1, self.edge2, ov) / d;
    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 || u + v > 1.0 || t < kEPS) {
        return HitRecord(false, vec3(0.0), vec3(0.0));
    }

    if (self.smooth_normal) {
        vec3 normal = normalize(self.normal0 * (1.0 - u - v) + self.normal1 * u + self.normal2 * v);
        return HitRecord(true, normal, self.vertex + self.edge1 * u + self.edge2 * v);
    }

    return HitRecord(true, normalize(cross(self.edge1, self.edge2)), self.vertex + self.edge1 * u + self.edge2 * v);
}

uniform int n_triangles;
uniform int n_materials;

const int textureSize = 1024;
Triangle fetchTriangle(int index) {
    int size = 24 / 4;
    int x = (index * size) % textureSize;
    int y = (index * size) / textureSize;

    vec3 vertex = texture(triangles_texture, vec2(float(x) / float(textureSize), float(y) / float(textureSize))).xyz;
    float material_id = texture(triangles_texture, vec2(float(x) / float(textureSize), float(y) / float(textureSize))).w;
    vec3 edge1 = texture(triangles_texture, vec2(float(x + 1) / float(textureSize), float(y) / float(textureSize))).xyz;
    vec3 edge2 = texture(triangles_texture, vec2(float(x + 2) / float(textureSize), float(y) / float(textureSize))).xyz;
    float smooth_normal = texture(triangles_texture, vec2(float(x + 2) / float(textureSize), float(y) / float(textureSize))).w;
    vec3 normal0 = texture(triangles_texture, vec2(float(x + 3) / float(textureSize), float(y) / float(textureSize))).xyz;
    vec3 normal1 = texture(triangles_texture, vec2(float(x + 4) / float(textureSize), float(y) / float(textureSize))).xyz;
    vec3 normal2 = texture(triangles_texture, vec2(float(x + 5) / float(textureSize), float(y) / float(textureSize))).xyz;

    return Triangle(vertex, edge1, edge2, normal0, normal1, normal2, int(material_id), smooth_normal > 0.0);
}

struct AABB {
    vec3 minv;
    vec3 maxv;
};

bool AABB_intersect(AABB self, Ray ray) {
    float tmin = (self.minv.x - ray.origin.x) / ray.direction.x;
    float tmax = (self.maxv.x - ray.origin.x) / ray.direction.x;

    if (tmin > tmax) {
        float tmp = tmin;
        tmin = tmax;
        tmax = tmp;
    }

    float tymin = (self.minv.y - ray.origin.y) / ray.direction.y;
    float tymax = (self.maxv.y - ray.origin.y) / ray.direction.y;

    if (tymin > tymax) {
        float tmp = tymin;
        tymin = tymax;
        tymax = tmp;
    }

    if ((tmin > tymax) || (tymin > tmax)) {
        return false;
    }

    if (tymin > tmin) {
        tmin = tymin;
    }

    if (tymax < tmax) {
        tmax = tymax;
    }

    float tzmin = (self.minv.z - ray.origin.z) / ray.direction.z;
    float tzmax = (self.maxv.z - ray.origin.z) / ray.direction.z;

    if (tzmin > tzmax) {
        float tmp = tzmin;
        tzmin = tzmax;
        tzmax = tmp;
    }

    if ((tmin > tzmax) || (tzmin > tmax)) {
        return false;
    }

    return true;
}

struct Material {
    int id;
    vec3 color;
    vec3 emission;
    vec3 specular;
    float specular_weight;
    AABB aabb;
    int t_index_min;
    int t_index_max;
};

Material fetchMaterial(int index) {
    int size = 20 / 4;
    int x = (index * size) % textureSize;
    int y = (index * size) / textureSize;

    vec3 color = texture(material_texture, vec2(float(x) / float(textureSize), float(y) / float(textureSize))).xyz;
    vec3 emission = texture(material_texture, vec2(float(x + 1) / float(textureSize), float(y) / float(textureSize))).xyz;
    vec3 specular = texture(material_texture, vec2(float(x + 2) / float(textureSize), float(y) / float(textureSize))).xyz;
    float specular_weight = texture(material_texture, vec2(float(x + 2) / float(textureSize), float(y) / float(textureSize))).w;
    vec3 minv = texture(material_texture, vec2(float(x + 3) / float(textureSize), float(y) / float(textureSize))).xyz;
    int t_index_min = int(texture(material_texture, vec2(float(x + 3) / float(textureSize), float(y) / float(textureSize))).w);
    vec3 maxv = texture(material_texture, vec2(float(x + 4) / float(textureSize), float(y) / float(textureSize))).xyz;
    int t_index_max = int(texture(material_texture, vec2(float(x + 4) / float(textureSize), float(y) / float(textureSize))).w);

    return Material(index, color, emission, specular, specular_weight, AABB(minv, maxv), t_index_min, t_index_max);
}

struct BVHTreeNode {
    uint bvh_tree_node_type;
    AABB aabb;
    int left;
    int right;
    int n_triangles;
    int t_index;
};

const uint BVHTreeNodeTypeNode = 0u;
const uint BVHTreeNodeTypeLeaf = 1u;

BVHTreeNode fetchBVHTreeNode(int index) {
    int x = index % textureSize;
    int y = index / textureSize;

    int cursor = int(texture(bvh_tree_texture, vec2(float(x) / float(textureSize), float(y) / float(textureSize))).x);

    int cx = cursor % textureSize;
    int cy = cursor / textureSize;

    vec3 minv = texture(bvh_tree_texture, vec2(float(cx) / float(textureSize), float(cy) / float(textureSize))).xyz;
    uint bvh_tree_node_type = uint(texture(bvh_tree_texture, vec2(float(cx) / float(textureSize), float(cy) / float(textureSize))).w);
    vec3 maxv = texture(bvh_tree_texture, vec2(float(cx + 1) / float(textureSize), float(cy) / float(textureSize))).xyz;
    int n_triangles = int(texture(bvh_tree_texture, vec2(float(cx + 1) / float(textureSize), float(cy) / float(textureSize))).w);

    return BVHTreeNode(
        bvh_tree_node_type,
        AABB(minv, maxv),
        index * 2 + 1,
        index * 2 + 2,
        n_triangles,
        cursor + 2
    );
}

const uint TTriangle = 0u;

struct HitInScene {
    int index;
    uint type;
    HitRecord r;
};

HitInScene intersect(Ray ray){
    float dist = 1000000.0;
    HitInScene hit = HitInScene(-1, TTriangle, HitRecord(false, vec3(0.0), vec3(0.0)));
    for(int i = 0; i < n_materials; i++){
        Material m = fetchMaterial(i);
        if (!AABB_intersect(m.aabb, ray)) {
            continue;
        }

        for(int i = m.t_index_min; i < m.t_index_max; i++){
            Triangle obj = fetchTriangle(i);
            if (obj.material_id != m.id) {
                continue;
            }
            HitRecord r = Triangle_intersect(obj, ray);

            if (r.hit) {
                float t = length(r.point - ray.origin);
                if (t < dist) {
                    dist = t;
                    hit.index = i;
                    hit.type = TTriangle;
                    hit.r = r;

                    continue;
                }
            }
        }
    }

    return hit;
}

void next_ray(HitInScene hit, float seed, inout Ray ray, out vec3 weight_delta) {
    vec3 object_color = vec3(1.0);
    if (hit.type == TTriangle) {
        Triangle t = fetchTriangle(hit.index);
        Material m = fetchMaterial(t.material_id);
        object_color = m.color;
    } else {
        object_color = vec3(1, 0, 1);
    }

    vec3 orienting_normal = dot(hit.r.normal, ray.direction) < 0.0 ? hit.r.normal : -hit.r.normal;
    weight_delta = object_color;
    ray.origin = hit.r.point + orienting_normal * kEPS;

    if (hit.type == TTriangle) {
        Triangle t = fetchTriangle(hit.index);
        Material m = fetchMaterial(t.material_id);

        if (m.specular_weight > 0.0) {
            float specular_prob = m.specular_weight / (m.specular_weight + 1.0);
            float r = rand(vec2(seed, m.specular_weight) + hit.r.point.xy);
            if (r < specular_prob) {
                ray.direction = reflect(ray.direction, orienting_normal);

                weight_delta = m.specular / specular_prob;
            } else {
                ray.direction = randOnHemisphere(orienting_normal, seed);

                weight_delta = object_color / (1.0 - specular_prob);
            }
        } else {
            ray.direction = randOnHemisphere(orienting_normal, seed);
        }
    } else {
        ray.direction = randOnHemisphere(orienting_normal, seed);
    }
}

vec3 raytrace(Ray ray) {
    vec3 color = vec3(0.0);
    vec3 weight = vec3(1.0);
    int count = 0;

    while (true) {
        HitInScene hit = intersect(ray);
        if (hit.index == -1) {
            return color;
        }

        vec3 object_color = vec3(1.0);
        if (hit.type == TTriangle) {
            Triangle t = fetchTriangle(hit.index);
            Material m = fetchMaterial(t.material_id);
            object_color = m.color;
        } else {
            object_color = vec3(1, 0, 1);
        }

        if (render_type == RenderTypeColor && hit.index != -1) {
            return object_color;
        }

        vec3 orienting_normal = dot(hit.r.normal, ray.direction) < 0.0 ? hit.r.normal : -hit.r.normal;

        if (render_type == RenderTypeNormal) {
            return orienting_normal + vec3(0.25);
        }

        if (hit.type == TTriangle) {
            Triangle t = fetchTriangle(hit.index);
            Material m = fetchMaterial(t.material_id);
            color += m.emission * weight;
        }

        float russian_roulette_threshold = 0.5;
        if (count < 5) {
            russian_roulette_threshold = 1.0;
        }
        if (count > 20) {
            russian_roulette_threshold *= pow(0.5, float(count - 5));
        }

        float seed = float(iterations) + float(count) + rand(hit.r.point.xy);
        float r = rand(vec2(seed, 0.0));
        if (r >= russian_roulette_threshold) {
            return color;
        }

        vec3 weight_delta = vec3(1.0);
        next_ray(hit, seed, ray, weight_delta);
        weight *= weight_delta / russian_roulette_threshold;
        count++;
    }
}

struct Camera {
    vec3 origin;
    vec3 up;
    vec3 direction;
    float screen_dist;
};

void main(void){
    Camera camera = Camera(camera_position, normalize(camera_up), normalize(camera_direction), screen_dist);
    float screen_width = 3.0;
    float screen_height = 3.0;

    vec3 screen_x = normalize(cross(camera.direction, camera.up)) * screen_width;
    vec3 screen_y = normalize(cross(screen_x, camera.direction)) * screen_height;
    vec3 screen_origin = camera.origin + camera.direction * camera.screen_dist;

    vec3 color = vec3(0.0);
    for (int i = 0; i < spp; i++) {
        vec2 dp = rand3(vec3(gl_FragCoord.xy + vec2(float(iterations)), float(i))).xy;
        vec2 p = (((gl_FragCoord.xy + dp - vec2(0.5)) * 2.0) - resolution.xy) / min(resolution.x, resolution.y);

        vec3 screen_p = screen_origin + screen_x * p.x + screen_y * p.y;
        Ray ray = Ray(camera.origin, normalize(screen_p - camera.origin));

        color += raytrace(ray);
    }

    vec4 prev = texture(u_texture, v_texcoord);
    vec4 current = vec4(color / float(spp), 1.0);

    outColor = prev + current;
}
