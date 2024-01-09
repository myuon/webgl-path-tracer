import shaderVertSource from "./glsl/shader.vert?raw";
import shaderFragSource from "./glsl/shader.frag?raw";
import rendererVertSource from "./glsl/renderer.vert?raw";
import rendererFragSource from "./glsl/renderer.frag?raw";
import GUI from "lil-gui";
import Stats from "stats.js";
import {
  loadMat4,
  loadMitsubaScene,
  loadMtlScene,
  loadObjScene,
} from "./scene";
import { vec3, vec4 } from "gl-matrix";
import { createProgramFromSource, createVao, diagnoseGlError } from "./webgl";

const renderTypes = ["render", "color", "normal"];

const objFiles = import.meta.glob("./scenes/cornell-box-mtl/*", { as: "raw" });
const xmlFiles = import.meta.glob("./scenes/**/*.xml", { as: "raw" });
const veachBidirModels = Object.fromEntries(
  Object.entries(
    import.meta.glob("./scenes/veach-bidir/**/*.obj", { as: "raw" })
  ).map(([k, v]) => {
    return ["models" + k.split("models")[1], v] as [
      string,
      () => Promise<string>
    ];
  })
);

const scenes = [
  ...Object.keys(xmlFiles).filter((t) => t.endsWith(".xml")),
  ...Object.keys(objFiles).filter((t) => t.endsWith(".obj")),
];

const loadSettings = (): {
  tick: boolean;
  renderType: string;
  spp: number;
  scene: string;
  renderBoundingBoxes: boolean;
} => {
  const def = {
    tick: true,
    renderType: "render",
    spp: 1,
    scene: "./scenes/cornell-box-mtl/CornellBox-Original.obj",
    renderBoundingBoxes: false,
  };
  try {
    const settings = localStorage.getItem("settings");

    return {
      ...def,
      ...JSON.parse(settings ?? ""),
    };
  } catch (err) {
    return def;
  }
};
const saveSettings = (value: any) => {
  localStorage.setItem("settings", JSON.stringify(value));
};

type BVHTree =
  | {
      type: "node";
      aabb: [vec3, vec3];
      left?: BVHTree;
      right?: BVHTree;
    }
  | {
      type: "leaf";
      aabb: [vec3, vec3];
      triangles: number[];
    };
type BVHShape = {
  id: number;
  vertex: vec3;
  edge1: vec3;
  edge2: vec3;
};

const constructBVHTree = (shapes: BVHShape[], depth: number): BVHTree => {
  console.log("constructBVHTree", depth);
  const constructMinimumAABB = (shapes: BVHShape[]) => {
    const aabb = [
      vec3.fromValues(Infinity, Infinity, Infinity),
      vec3.fromValues(-Infinity, -Infinity, -Infinity),
    ] as [vec3, vec3];

    for (let i = 0; i < shapes.length; i++) {
      appendAABB(aabb, shapes[i]);
    }

    return aabb;
  };
  const surfaceAABB = (aabb: [vec3, vec3]) => {
    const size = vec3.create();
    vec3.subtract(size, aabb[1], aabb[0]);
    return 2 * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0]);
  };
  const appendAABB = (aabb: [vec3, vec3], shape: BVHShape) => {
    const area = vec3.create();
    vec3.cross(area, shape.edge1, shape.edge2);
    if (vec3.length(area) > 0.5) {
      return aabb;
    }

    vec3.min(aabb[0], aabb[0], shape.vertex);
    vec3.max(aabb[1], aabb[1], shape.vertex);

    let v2 = vec3.create();
    vec3.add(v2, shape.vertex, shape.edge1);
    vec3.min(aabb[0], aabb[0], v2);
    vec3.max(aabb[1], aabb[1], v2);

    let v3 = vec3.create();
    vec3.add(v3, shape.vertex, shape.edge2);
    vec3.min(aabb[0], aabb[0], v3);
    vec3.max(aabb[1], aabb[1], v3);

    return aabb;
  };

  const COST_AABB = 1;
  const COST_TRIANGLE = 2;

  const boxAll = constructMinimumAABB(shapes);

  let bestCost = COST_TRIANGLE * shapes.length;
  let bestLeft: BVHShape[] = [];
  let bestRight: BVHShape[] = [];

  let result: BVHTree | undefined = undefined;

  const sortShapes = (axisIndex: number) =>
    shapes.sort((a, b) => {
      const aCenter = vec3.create();
      vec3.add(aCenter, a.edge1, a.edge2);
      vec3.scale(aCenter, aCenter, 1 / 3);
      vec3.add(aCenter, aCenter, a.vertex);

      const bCenter = vec3.create();
      vec3.add(bCenter, b.edge1, b.edge2);
      vec3.scale(bCenter, bCenter, 1 / 3);
      vec3.add(bCenter, bCenter, b.vertex);

      return aCenter[axisIndex] - bCenter[axisIndex];
    });
  const sortedShapes = [sortShapes(0), sortShapes(1), sortShapes(2)];

  // split
  ["x", "y", "z"].forEach((_, axisIndex) => {
    let prevBoxLeft = [
      vec3.fromValues(Infinity, Infinity, Infinity),
      vec3.fromValues(-Infinity, -Infinity, -Infinity),
    ] as [vec3, vec3];

    const boxRightSurfaces = Array.from(
      { length: sortedShapes[axisIndex].length },
      () => 0.0
    );
    let prevBoxRight = [
      vec3.fromValues(Infinity, Infinity, Infinity),
      vec3.fromValues(-Infinity, -Infinity, -Infinity),
    ] as [vec3, vec3];
    [...sortedShapes[axisIndex]].reverse().forEach((shape, i) => {
      const boxRight = appendAABB(prevBoxRight, shape);
      boxRightSurfaces[sortedShapes[axisIndex].length - i - 1] =
        surfaceAABB(boxRight);
      prevBoxRight = boxRight;
    });
    sortedShapes[axisIndex].forEach((shape, splitAt) => {
      const boxLeft = appendAABB(prevBoxLeft, shape);
      prevBoxLeft = boxLeft;

      const cost =
        COST_AABB * 2 +
        (COST_TRIANGLE * shapes.length * surfaceAABB(boxLeft)) /
          surfaceAABB(boxAll) +
        (COST_TRIANGLE * shapes.length * boxRightSurfaces[splitAt]) /
          surfaceAABB(boxAll);
      if (cost < bestCost) {
        result = {
          type: "node" as const,
          aabb: boxAll,
          left: undefined,
          right: undefined,
        };
        bestLeft = sortedShapes[axisIndex].slice(0, splitAt);
        bestRight = sortedShapes[axisIndex].slice(splitAt);
        bestCost = cost;
      }
    });
  });
  if (result === undefined) {
    result = {
      type: "leaf",
      aabb: boxAll,
      triangles: shapes.map((t) => t.id),
    };
  }

  const resultAsTree = result as BVHTree;
  if (resultAsTree.type === "node") {
    resultAsTree.left = constructBVHTree(bestLeft, depth + 1);
    resultAsTree.right = constructBVHTree(bestRight, depth + 1);
  }

  return result;
};

const createTrianglesFromAABB = (
  aabb: [vec3, vec3]
): { vertex: vec3; edge1: vec3; edge2: vec3 }[] => {
  const [a, b] = aabb;

  const dx = vec3.fromValues(b[0] - a[0], 0, 0);
  const dy = vec3.fromValues(0, b[1] - a[1], 0);
  const dz = vec3.fromValues(0, 0, b[2] - a[2]);
  const dxdy = vec3.fromValues(b[0] - a[0], b[1] - a[1], 0);
  const dxdz = vec3.fromValues(b[0] - a[0], 0, b[2] - a[2]);
  const dydz = vec3.fromValues(0, b[1] - a[1], b[2] - a[2]);

  return [
    {
      vertex: a,
      edge1: dx,
      edge2: dxdy,
    },
    {
      vertex: a,
      edge1: dxdy,
      edge2: dy,
    },
    {
      vertex: a,
      edge1: dy,
      edge2: dydz,
    },
    {
      vertex: a,
      edge1: dydz,
      edge2: dz,
    },
    {
      vertex: a,
      edge1: dz,
      edge2: dxdz,
    },
    {
      vertex: a,
      edge1: dxdz,
      edge2: dx,
    },
    {
      vertex: [a[0], a[1], b[2]],
      edge1: dx,
      edge2: dxdy,
    },
    {
      vertex: [a[0], a[1], b[2]],
      edge1: dxdy,
      edge2: dy,
    },
    {
      vertex: [a[0], b[1], a[2]],
      edge1: dx,
      edge2: dxdz,
    },
    {
      vertex: [a[0], b[1], a[2]],
      edge1: dxdz,
      edge2: dz,
    },
    {
      vertex: [b[0], a[1], a[2]],
      edge1: dy,
      edge2: dydz,
    },
    {
      vertex: [b[0], a[1], a[2]],
      edge1: dydz,
      edge2: dz,
    },
  ];
};

const loadScene = async (
  sceneFile: string,
  renderBoundingBoxes: boolean,
  gl: WebGL2RenderingContext
) => {
  let camera: {
    position: vec3;
    up: vec3;
    direction: vec3;
    screen_dist: number;
  };

  const materials: Record<
    string,
    {
      id: number;
      name: string;
      emission: vec3;
      color: vec3;
      specular: vec3;
      specularWeight: number;
      aabb?: [vec3, vec3];
      triangles?: [number, number];
    }
  > = {};

  const triangles: {
    id: number;
    type: "triangle";
    triangle: {
      vertex: vec3;
      edge1: vec3;
      edge2: vec3;
      normal1?: vec3;
      normal2?: vec3;
      normal3?: vec3;
    };
    materialId: number;
    smooth: boolean;
  }[] = [];

  if (sceneFile.endsWith(".xml")) {
    const scene = await loadMitsubaScene(
      await xmlFiles[sceneFile](),
      sceneFile === "./scenes/veach-bidir/scene.xml" ? veachBidirModels : {}
    );

    camera = {
      ...scene.sensors.camera,
      screen_dist: 3.0 / Math.tan((scene.sensors.fov / 2) * (Math.PI / 180)),
    };
    console.log(camera);

    scene.shapes.forEach((shape) => {
      if (shape.type === "rectangle") {
        const aabb = [
          vec3.fromValues(Infinity, Infinity, Infinity),
          vec3.fromValues(-Infinity, -Infinity, -Infinity),
        ] as [vec3, vec3];
        const mat = loadMat4(shape.matrix);
        const plane = [
          [-1, 1, 0, 1],
          [-1, -1, 0, 1],
          [1, -1, 0, 1],
          [1, 1, 0, 1],
        ] as [number, number, number, number][];

        const p1 = vec4.create();
        vec4.transformMat4(p1, plane[0], mat);

        const p2 = vec4.create();
        vec4.transformMat4(p2, plane[1], mat);

        const p3 = vec4.create();
        vec4.transformMat4(p3, plane[2], mat);

        const p4 = vec4.create();
        vec4.transformMat4(p4, plane[3], mat);

        const p12 = vec4.create();
        vec4.subtract(p12, p2, p1);

        const p13 = vec4.create();
        vec4.subtract(p13, p3, p1);

        const p14 = vec4.create();
        vec4.subtract(p14, p4, p1);

        const materialId = Object.keys(materials).length;

        triangles.push({
          id: triangles.length,
          type: "triangle",
          triangle: {
            vertex: [p1[0], p1[1], p1[2]],
            edge1: [p12[0], p12[1], p12[2]],
            edge2: [p13[0], p13[1], p13[2]],
          },
          materialId,
          smooth: false,
        });
        triangles.push({
          id: triangles.length,
          type: "triangle",
          triangle: {
            vertex: [p1[0], p1[1], p1[2]],
            edge1: [p14[0], p14[1], p14[2]],
            edge2: [p13[0], p13[1], p13[2]],
          },
          materialId,
          smooth: false,
        });

        vec3.min(aabb[0], aabb[0], [p1[0], p1[1], p1[2]]);
        vec3.max(aabb[1], aabb[1], [p1[0], p1[1], p1[2]]);
        vec3.min(aabb[0], aabb[0], [p2[0], p2[1], p2[2]]);
        vec3.max(aabb[1], aabb[1], [p2[0], p2[1], p2[2]]);
        vec3.min(aabb[0], aabb[0], [p3[0], p3[1], p3[2]]);
        vec3.max(aabb[1], aabb[1], [p3[0], p3[1], p3[2]]);
        vec3.min(aabb[0], aabb[0], [p4[0], p4[1], p4[2]]);
        vec3.max(aabb[1], aabb[1], [p4[0], p4[1], p4[2]]);

        materials[shape.id] = {
          id: materialId,
          name: shape.id,
          emission: [
            shape.emitter?.radiance[0] ?? 0.0,
            shape.emitter?.radiance[1] ?? 0.0,
            shape.emitter?.radiance[2] ?? 0.0,
          ],
          color: shape.bsdf?.reflectance ?? [0.0, 0.0, 0.0],
          specular: [0.0, 0.0, 0.0],
          specularWeight: 0.0,
          aabb,
          triangles: [triangles.length - 2, triangles.length],
        };
      } else if (shape.type === "cube") {
        const aabb = [
          vec3.fromValues(Infinity, Infinity, Infinity),
          vec3.fromValues(-Infinity, -Infinity, -Infinity),
        ] as [vec3, vec3];

        const planes = [
          [
            [-1, 1, -1, 1],
            [-1, -1, -1, 1],
            [1, -1, -1, 1],
            [1, 1, -1, 1],
          ],
          [
            [-1, 1, 1, 1],
            [-1, -1, 1, 1],
            [1, -1, 1, 1],
            [1, 1, 1, 1],
          ],
          [
            [-1, 1, 1, 1],
            [-1, 1, -1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, 1],
          ],
          [
            [-1, -1, 1, 1],
            [-1, -1, -1, 1],
            [1, -1, -1, 1],
            [1, -1, 1, 1],
          ],
          [
            [1, 1, -1, 1],
            [1, -1, -1, 1],
            [1, -1, 1, 1],
            [1, 1, 1, 1],
          ],
          [
            [-1, 1, -1, 1],
            [-1, -1, -1, 1],
            [-1, -1, 1, 1],
            [-1, 1, 1, 1],
          ],
        ] as [number, number, number, number][][];

        const materialId = Object.keys(materials).length;

        planes.forEach((plane) => {
          const mat = loadMat4(shape.matrix);

          const p1 = vec4.create();
          vec4.transformMat4(p1, plane[0], mat);

          const p2 = vec4.create();
          vec4.transformMat4(p2, plane[1], mat);

          const p3 = vec4.create();
          vec4.transformMat4(p3, plane[2], mat);

          const p4 = vec4.create();
          vec4.transformMat4(p4, plane[3], mat);

          const p12 = vec4.create();
          vec4.subtract(p12, p2, p1);

          const p13 = vec4.create();
          vec4.subtract(p13, p3, p1);

          const p14 = vec4.create();
          vec4.subtract(p14, p4, p1);

          triangles.push({
            id: triangles.length,
            type: "triangle",
            triangle: {
              vertex: [p1[0], p1[1], p1[2]],
              edge1: [p12[0], p12[1], p12[2]],
              edge2: [p13[0], p13[1], p13[2]],
            },
            materialId,
            smooth: false,
          });
          triangles.push({
            id: triangles.length,
            type: "triangle",
            triangle: {
              vertex: [p1[0], p1[1], p1[2]],
              edge1: [p14[0], p14[1], p14[2]],
              edge2: [p13[0], p13[1], p13[2]],
            },
            materialId,
            smooth: false,
          });

          vec3.min(aabb[0], aabb[0], [p1[0], p1[1], p1[2]]);
          vec3.max(aabb[1], aabb[1], [p1[0], p1[1], p1[2]]);
          vec3.min(aabb[0], aabb[0], [p2[0], p2[1], p2[2]]);
          vec3.max(aabb[1], aabb[1], [p2[0], p2[1], p2[2]]);
          vec3.min(aabb[0], aabb[0], [p3[0], p3[1], p3[2]]);
          vec3.max(aabb[1], aabb[1], [p3[0], p3[1], p3[2]]);
          vec3.min(aabb[0], aabb[0], [p4[0], p4[1], p4[2]]);
          vec3.max(aabb[1], aabb[1], [p4[0], p4[1], p4[2]]);
        });

        materials[shape.id] = {
          id: materialId,
          name: shape.id,
          emission: [
            shape.emitter?.radiance[0] ?? 0.0,
            shape.emitter?.radiance[1] ?? 0.0,
            shape.emitter?.radiance[2] ?? 0.0,
          ],
          color: shape.bsdf?.reflectance ?? [0.0, 0.0, 0.0],
          specular: [0.0, 0.0, 0.0],
          specularWeight: 0.0,
          aabb,
          triangles: [triangles.length - 12, triangles.length],
        };
      } else if (shape.type === "obj") {
        const aabb = [
          vec3.fromValues(Infinity, Infinity, Infinity),
          vec3.fromValues(-Infinity, -Infinity, -Infinity),
        ] as [vec3, vec3];

        const materialId = Object.keys(materials).length;

        let trianglesIndexStart = triangles.length;

        shape.model?.forEach((triangle) => {
          let e1 = vec3.create();
          vec3.subtract(e1, triangle.vertices[1], triangle.vertices[0]);

          let e2 = vec3.create();
          vec3.subtract(e2, triangle.vertices[2], triangle.vertices[0]);

          if (!renderBoundingBoxes) {
            triangles.push({
              id: triangles.length,
              type: "triangle",
              triangle: {
                vertex: triangle.vertices[0],
                edge1: e1,
                edge2: e2,
                normal1: triangle.normals[0],
                normal2: triangle.normals[1],
                normal3: triangle.normals[2],
              },
              materialId,
              smooth: true,
            });
          }

          vec3.min(aabb[0], aabb[0], triangle.vertices[0]);
          vec3.max(aabb[1], aabb[1], triangle.vertices[0]);
          vec3.min(aabb[0], aabb[0], triangle.vertices[1]);
          vec3.max(aabb[1], aabb[1], triangle.vertices[1]);
          vec3.min(aabb[0], aabb[0], triangle.vertices[2]);
          vec3.max(aabb[1], aabb[1], triangle.vertices[2]);
        });

        if (renderBoundingBoxes) {
          createTrianglesFromAABB(aabb).forEach((triangle) => {
            triangles.push({
              id: triangles.length,
              type: "triangle",
              triangle,
              materialId,
              smooth: false,
            });
          });
        }

        materials[shape.id] = {
          id: materialId,
          name: shape.id,
          emission: [
            shape.emitter?.radiance[0] ?? 0.0,
            shape.emitter?.radiance[1] ?? 0.0,
            shape.emitter?.radiance[2] ?? 0.0,
          ],
          color: renderBoundingBoxes
            ? [1.0, 0.0, 1.0]
            : shape.bsdf?.reflectance ?? [0.0, 0.0, 0.0],
          specular: [0.0, 0.0, 0.0],
          specularWeight: 0.0,
          aabb,
          triangles: [trianglesIndexStart, triangles.length],
        };
      } else {
        console.warn(
          `Unknown shape type: ${shape.type} (${JSON.stringify(shape)})`
        );
      }
    });
    console.log(scene);
  } else {
    const boxObj = loadObjScene(await objFiles[sceneFile]());
    const boxMtl = loadMtlScene(
      await objFiles[sceneFile.replace(".obj", ".mtl")]()
    );
    console.log(boxObj);
    console.log(boxMtl);

    let up = vec3.create();
    vec3.normalize(up, [0.0, 1.0, 0.0]);

    let dir = vec3.create();
    vec3.normalize(dir, [0.0, 0.0, -1.0]);

    camera = {
      position: vec3.fromValues(0.0, 1.0, 5.0),
      up,
      direction: dir,
      screen_dist: 8,
    };
    console.log(camera);

    boxObj.objects.forEach((object) => {
      const fs = object.faces;
      const aabb = [
        vec3.fromValues(Infinity, Infinity, Infinity),
        vec3.fromValues(-Infinity, -Infinity, -Infinity),
      ] as [vec3, vec3];
      const trianglesIndexStart = triangles.length;

      fs.forEach((f) => {
        if (f.vertices.length === 3) {
          let e10 = vec3.create();
          vec3.subtract(e10, f.vertices[1], f.vertices[0]);

          let e20 = vec3.create();
          vec3.subtract(e20, f.vertices[2], f.vertices[0]);

          const material = boxMtl.find((m) => m.name === object.usemtl)!;
          const materialId = materials[object.usemtl]
            ? materials[object.usemtl].id
            : Object.keys(materials).length;
          materials[object.usemtl] = {
            id: materialId,
            name: object.usemtl,
            emission: material.Ke ?? [0.0, 0.0, 0.0],
            color: material.Ka ?? [0.0, 0.0, 0.0],
            specular: material.Ks ?? [0.0, 0.0, 0.0],
            specularWeight: material.Ns ?? 0.0,
          };

          triangles.push({
            id: triangles.length,
            type: "triangle",
            triangle: {
              vertex: f.vertices[0],
              edge1: e10,
              edge2: e20,
              normal1: f.normals[0],
              normal2: f.normals[1],
              normal3: f.normals[2],
            },
            materialId,
            smooth: object.smooth ?? false,
          });

          vec3.min(aabb[0], aabb[0], f.vertices[0]);
          vec3.max(aabb[1], aabb[1], f.vertices[0]);
          vec3.min(aabb[0], aabb[0], f.vertices[1]);
          vec3.max(aabb[1], aabb[1], f.vertices[1]);
          vec3.min(aabb[0], aabb[0], f.vertices[2]);
          vec3.max(aabb[1], aabb[1], f.vertices[2]);
        } else if (f.vertices.length === 4) {
          let e10 = vec3.create();
          vec3.subtract(e10, f.vertices[1], f.vertices[0]);

          let e20 = vec3.create();
          vec3.subtract(e20, f.vertices[2], f.vertices[0]);

          let e30 = vec3.create();
          vec3.subtract(e30, f.vertices[3], f.vertices[0]);

          const material = boxMtl.find((m) => m.name === object.usemtl)!;
          const materialId = materials[object.usemtl]
            ? materials[object.usemtl].id
            : Object.keys(materials).length;
          materials[object.usemtl] = {
            id: materialId,
            name: object.usemtl,
            emission: material.Ke ?? [0.0, 0.0, 0.0],
            color: material.Ka ?? [0.0, 0.0, 0.0],
            specular: material.Ks ?? [0.0, 0.0, 0.0],
            specularWeight: material.Ns ?? 0.0,
          };

          triangles.push({
            id: triangles.length,
            type: "triangle",
            triangle: {
              vertex: f.vertices[0],
              edge1: e10,
              edge2: e20,
            },
            materialId,
            smooth: object.smooth ?? false,
          });
          triangles.push({
            id: triangles.length,
            type: "triangle",
            triangle: {
              vertex: f.vertices[0],
              edge1: e30,
              edge2: e20,
            },
            materialId,
            smooth: object.smooth ?? false,
          });
        } else {
          console.error("not implemented");
          throw new Error("not implemented");
        }
      });

      const trianglesIndexEnd = triangles.length;

      materials[object.usemtl].aabb = aabb;
      materials[object.usemtl].triangles = [
        trianglesIndexStart,
        trianglesIndexEnd,
      ];
    });
  }
  console.log(triangles);
  console.log(materials);

  const now = performance.now();

  const textureSize = 1024;

  const bvhTree = constructBVHTree(
    triangles.map((t) => ({ id: t.id, ...t.triangle })),
    0
  );
  console.log("Tree constructed (sec):", (performance.now() - now) / 1000);
  console.log(bvhTree);

  const bvhTreeTextureData = new Float32Array(textureSize * textureSize * 4);
  const getMaxTreeIndex = (tree: BVHTree, position: number): number => {
    if (tree.type === "node") {
      return Math.max(
        getMaxTreeIndex(tree.left!, 2 * position + 1),
        getMaxTreeIndex(tree.right!, 2 * position + 2)
      );
    } else if (tree.type === "leaf") {
      return position;
    } else {
      throw new Error("Unknown tree type");
    }
  };
  const serializeBVHTree = (
    tree: BVHTree,
    treeIndex: number,
    nodeCursor: number
  ): number => {
    if (tree.type === "node") {
      let cursor = nodeCursor;
      bvhTreeTextureData[treeIndex] = cursor;

      bvhTreeTextureData[cursor + 0] = tree.aabb[0][0];
      bvhTreeTextureData[cursor + 1] = tree.aabb[0][1];
      bvhTreeTextureData[cursor + 2] = tree.aabb[0][2];
      bvhTreeTextureData[cursor + 3] = 0; // type: "node"

      bvhTreeTextureData[cursor + 4] = tree.aabb[1][0];
      bvhTreeTextureData[cursor + 5] = tree.aabb[1][1];
      bvhTreeTextureData[cursor + 6] = tree.aabb[1][2];

      cursor += 8;

      cursor = serializeBVHTree(tree.left!, 2 * treeIndex + 1, cursor);
      cursor = serializeBVHTree(tree.right!, 2 * treeIndex + 2, cursor);

      return cursor;
    } else if (tree.type === "leaf") {
      let cursor = nodeCursor;
      bvhTreeTextureData[treeIndex] = cursor;

      bvhTreeTextureData[cursor + 0] = tree.aabb[0][0];
      bvhTreeTextureData[cursor + 1] = tree.aabb[0][1];
      bvhTreeTextureData[cursor + 2] = tree.aabb[0][2];
      bvhTreeTextureData[cursor + 3] = 1; // type: "leaf"

      bvhTreeTextureData[cursor + 4] = tree.aabb[1][0];
      bvhTreeTextureData[cursor + 5] = tree.aabb[1][1];
      bvhTreeTextureData[cursor + 6] = tree.aabb[1][2];
      bvhTreeTextureData[cursor + 7] = tree.triangles.length;

      cursor += 8;

      tree.triangles.forEach((triangleId, i) => {
        bvhTreeTextureData[cursor + i * 4] = triangleId;
      });

      cursor += tree.triangles.length * 4;

      return cursor;
    } else {
      throw new Error("Unknown tree type");
    }
  };
  serializeBVHTree(bvhTree, 0, getMaxTreeIndex(bvhTree, 0));

  gl.activeTexture(gl.TEXTURE3);
  const bvhTreeTexture = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, bvhTreeTexture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA32F,
    textureSize,
    textureSize,
    0,
    gl.RGBA,
    gl.FLOAT,
    bvhTreeTextureData
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.bindTexture(gl.TEXTURE_2D, null);

  const materialId = Object.keys(materials).length;

  materials["bvh"] = {
    id: materialId,
    name: "bvh",
    emission: [0.0, 0.0, 0.0],
    color: [1.0, 0.0, 1.0],
    specular: [0.0, 0.0, 0.0],
    specularWeight: 0.0,
    aabb: bvhTree.aabb,
    triangles: [triangles.length - 12 * 3, triangles.length],
  };

  const triangleTextureData = new Float32Array(textureSize * textureSize * 4);
  triangles.forEach((triangle, i) => {
    const size = 24;

    triangleTextureData[i * size + 0] = triangle.triangle.vertex[0];
    triangleTextureData[i * size + 1] = triangle.triangle.vertex[1];
    triangleTextureData[i * size + 2] = triangle.triangle.vertex[2];
    triangleTextureData[i * size + 3] = triangle.materialId;

    triangleTextureData[i * size + 4] = triangle.triangle.edge1[0];
    triangleTextureData[i * size + 5] = triangle.triangle.edge1[1];
    triangleTextureData[i * size + 6] = triangle.triangle.edge1[2];

    triangleTextureData[i * size + 8] = triangle.triangle.edge2[0];
    triangleTextureData[i * size + 9] = triangle.triangle.edge2[1];
    triangleTextureData[i * size + 10] = triangle.triangle.edge2[2];
    triangleTextureData[i * size + 11] = triangle.smooth ? 1.0 : 0.0;

    triangleTextureData[i * size + 12] = triangle.triangle.normal1?.[0] ?? 0.0;
    triangleTextureData[i * size + 13] = triangle.triangle.normal1?.[1] ?? 0.0;
    triangleTextureData[i * size + 14] = triangle.triangle.normal1?.[2] ?? 0.0;

    triangleTextureData[i * size + 16] = triangle.triangle.normal2?.[0] ?? 0.0;
    triangleTextureData[i * size + 17] = triangle.triangle.normal2?.[1] ?? 0.0;
    triangleTextureData[i * size + 18] = triangle.triangle.normal2?.[2] ?? 0.0;

    triangleTextureData[i * size + 20] = triangle.triangle.normal3?.[0] ?? 0.0;
    triangleTextureData[i * size + 21] = triangle.triangle.normal3?.[1] ?? 0.0;
    triangleTextureData[i * size + 22] = triangle.triangle.normal3?.[2] ?? 0.0;
  });

  gl.activeTexture(gl.TEXTURE1);
  const triangleTexture = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, triangleTexture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA32F,
    textureSize,
    textureSize,
    0,
    gl.RGBA,
    gl.FLOAT,
    triangleTextureData
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.bindTexture(gl.TEXTURE_2D, null);

  const materialTextureData = new Float32Array(textureSize * textureSize * 4);
  Object.values(materials).forEach((material) => {
    const size = 20;

    materialTextureData[material.id * size + 0] = material.color[0];
    materialTextureData[material.id * size + 1] = material.color[1];
    materialTextureData[material.id * size + 2] = material.color[2];

    materialTextureData[material.id * size + 4] = material.emission[0];
    materialTextureData[material.id * size + 5] = material.emission[1];
    materialTextureData[material.id * size + 6] = material.emission[2];

    materialTextureData[material.id * size + 8] = material.specular[0];
    materialTextureData[material.id * size + 9] = material.specular[1];
    materialTextureData[material.id * size + 10] = material.specular[2];
    materialTextureData[material.id * size + 11] = material.specularWeight;

    materialTextureData[material.id * size + 12] = material.aabb![0][0];
    materialTextureData[material.id * size + 13] = material.aabb![0][1];
    materialTextureData[material.id * size + 14] = material.aabb![0][2];
    materialTextureData[material.id * size + 15] =
      material.triangles?.[0] ?? -1;

    materialTextureData[material.id * size + 16] = material.aabb![1][0];
    materialTextureData[material.id * size + 17] = material.aabb![1][1];
    materialTextureData[material.id * size + 18] = material.aabb![1][2];
    materialTextureData[material.id * size + 19] =
      material.triangles?.[1] ?? -1;
  });

  gl.activeTexture(gl.TEXTURE2);
  const materialTexture = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, materialTexture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA32F,
    textureSize,
    textureSize,
    0,
    gl.RGBA,
    gl.FLOAT,
    materialTextureData
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.bindTexture(gl.TEXTURE_2D, null);

  return {
    triangleTexture,
    materialTexture,
    bvhTreeTexture,
    camera,
    triangles,
    materials,
  };
};

const main = async () => {
  const output = document.getElementById("output")! as HTMLDivElement;
  const canvas = document.getElementById("glcanvas")! as HTMLCanvasElement;
  const gl = canvas.getContext("webgl2");
  if (!gl) {
    console.error("Failed to get WebGL context");
    return;
  }

  console.log(gl.getExtension("EXT_color_buffer_float")!);
  console.log(gl.getParameter(gl.MAX_COMBINED_UNIFORM_BLOCKS));
  console.log(gl.getParameter(gl.MAX_TEXTURE_SIZE));

  const textures = Array.from({ length: 2 }).map(() => {
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.activeTexture(gl.TEXTURE0);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA32F,
      canvas.width,
      canvas.height,
      0,
      gl.RGBA,
      gl.FLOAT,
      null
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return tex;
  });

  const value = loadSettings();
  let {
    triangleTexture,
    materialTexture,
    camera,
    triangles,
    materials,
    bvhTreeTexture,
  } = await loadScene(value.scene, value.renderBoundingBoxes, gl)!;

  const stats = new Stats();
  stats.showPanel(0);
  document.body.appendChild(stats.dom);

  let iterations = 1;

  const reset = () => {
    iterations = 1;

    // clear textures
    textures.forEach((tex) => {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.activeTexture(gl.TEXTURE0);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA32F,
        canvas.width,
        canvas.height,
        0,
        gl.RGBA,
        gl.FLOAT,
        null
      );
      gl.bindTexture(gl.TEXTURE_2D, null);
    });
  };

  const gui = new GUI();
  gui.add(value, "tick").onChange((v: boolean) => {
    value.tick = v;
    saveSettings(value);
  });
  gui
    .add(value, "spp", [1, 2, 4, 8, 16, 32, 64, 128, 256])
    .onChange((v: number) => {
      value.spp = v;
      reset();
      saveSettings(value);
    });
  gui.add(value, "renderType", renderTypes).onChange((v: string) => {
    value.renderType = v;
    reset();
    saveSettings(value);
  });
  gui.add({ reset }, "reset");
  gui.add(value, "scene", scenes).onChange(async (v: string) => {
    value.scene = v;
    saveSettings(value);

    const resp = await loadScene(v, value.renderBoundingBoxes, gl)!;
    triangleTexture = resp.triangleTexture;
    materialTexture = resp.materialTexture;
    camera = resp.camera;
    triangles = resp.triangles;
    materials = resp.materials;
    bvhTreeTexture = resp.bvhTreeTexture;

    reset();
  });
  gui.add(value, "renderBoundingBoxes").onChange(async (v: boolean) => {
    value.renderBoundingBoxes = v;
    saveSettings(value);

    const resp = await loadScene(value.scene, value.renderBoundingBoxes, gl)!;
    triangleTexture = resp.triangleTexture;
    materialTexture = resp.materialTexture;
    camera = resp.camera;
    triangles = resp.triangles;
    materials = resp.materials;
    bvhTreeTexture = resp.bvhTreeTexture;

    reset();
  });

  let angleX = 0.0;
  let angleY = 0.0;

  let prevMousePosition: [number, number] | null = null;
  canvas.addEventListener("mousedown", (e) => {
    prevMousePosition = [e.clientX, e.clientY];
  });
  canvas.addEventListener("mouseup", () => {
    prevMousePosition = null;
  });
  canvas.addEventListener("mousemove", (e) => {
    if (prevMousePosition) {
      const dx = e.clientX - prevMousePosition[0];
      const dy = e.clientY - prevMousePosition[1];

      angleX += dy * 0.01;
      angleY -= dx * 0.01;

      angleX = Math.max(Math.min(angleX, Math.PI / 2), -Math.PI / 2);
      angleY = angleY % (Math.PI * 2);

      const lookAt = vec3.create();
      vec3.scaleAndAdd(
        lookAt,
        camera.position,
        camera.direction,
        camera.screen_dist
      );

      let cam = vec3.create();
      vec3.subtract(cam, camera.position, lookAt);
      vec3.normalize(cam, cam);

      let phi = Math.atan2(cam[2], cam[0]);
      if (phi < 0) {
        phi += Math.PI * 2;
      } else if (phi > Math.PI * 2) {
        phi -= Math.PI * 2;
      }
      let theta = Math.acos(cam[1]);

      phi -= 0.01 * dx;
      theta += 0.01 * dy;

      cam = [
        Math.sin(theta) * Math.cos(phi),
        Math.cos(theta),
        Math.sin(theta) * Math.sin(phi),
      ];

      vec3.scaleAndAdd(camera.position, lookAt, cam, camera.screen_dist);

      camera.direction = [-cam[0], -cam[1], -cam[2]];

      const right = vec3.create();
      vec3.cross(right, camera.direction, [0, 1, 0]);

      vec3.cross(camera.up, right, camera.direction);

      prevMousePosition = [e.clientX, e.clientY];
      reset();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "w") {
      const up = vec3.fromValues(camera.up[0], camera.up[1], camera.up[2]);
      vec3.scale(up, up, 0.1);

      vec3.add(camera.position, camera.position, up);
      iterations = 1;
      reset();
    } else if (e.key === "s") {
      const up = vec3.fromValues(camera.up[0], camera.up[1], camera.up[2]);
      vec3.scale(up, up, 0.1);

      vec3.subtract(camera.position, camera.position, up);
      iterations = 1;
      reset();
    } else if (e.key === "a") {
      const right = vec3.create();
      vec3.cross(right, camera.direction, camera.up);
      vec3.scale(right, right, 0.1);

      vec3.subtract(camera.position, camera.position, right);
      iterations = 1;
      reset();
    } else if (e.key === "d") {
      const right = vec3.create();
      vec3.cross(right, camera.direction, camera.up);
      vec3.scale(right, right, 0.1);

      vec3.add(camera.position, camera.position, right);
      iterations = 1;
      reset();
    } else if (e.key == "q") {
      camera.position = [
        camera.position[0] - camera.direction[0],
        camera.position[1] - camera.direction[1],
        camera.position[2] - camera.direction[2],
      ];
      iterations = 1;
      reset();
    } else if (e.key == "e") {
      camera.position = [
        camera.position[0] + camera.direction[0],
        camera.position[1] + camera.direction[1],
        camera.position[2] + camera.direction[2],
      ];
      iterations = 1;
      reset();
    }
  });

  const program = createProgramFromSource(
    gl,
    shaderVertSource,
    shaderFragSource
  );
  if (!program) return;

  const programLocations = {
    position: gl.getAttribLocation(program, "position"),
    texcoord: gl.getAttribLocation(program, "a_texcoord"),
    texture: gl.getUniformLocation(program, "u_texture"),
    trianglesTexture: gl.getUniformLocation(program, "triangles_texture"),
    materialTexture: gl.getUniformLocation(program, "material_texture"),
    iterations: gl.getUniformLocation(program, "iterations"),
    resolution: gl.getUniformLocation(program, "resolution"),
    camera_position: gl.getUniformLocation(program, "camera_position"),
    camera_direction: gl.getUniformLocation(program, "camera_direction"),
    camera_up: gl.getUniformLocation(program, "camera_up"),
    screen_dist: gl.getUniformLocation(program, "screen_dist"),
    spp: gl.getUniformLocation(program, "spp"),
    n_triangles: gl.getUniformLocation(program, "n_triangles"),
    n_materials: gl.getUniformLocation(program, "n_materials"),
    render_type: gl.getUniformLocation(program, "render_type"),
    bvhTreeTexture: gl.getUniformLocation(program, "bvh_tree_texture"),
  };

  const shaderVao = createVao(
    gl,
    [
      [
        [-1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
      ].flat(),
      [
        [-1.0, 1.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
      ].flat(),
    ],
    [programLocations.position, programLocations.texcoord],
    [3, 2],
    [
      [0, 1, 2],
      [1, 2, 3],
    ].flat()
  );
  if (!shaderVao) {
    console.error("Failed to create vertexArray");
    return;
  }

  const rendererProgram = createProgramFromSource(
    gl,
    rendererVertSource,
    rendererFragSource
  );
  if (!rendererProgram) return;

  const rendererProgramLocations = {
    position: gl.getAttribLocation(rendererProgram, "position"),
    texcoord: gl.getAttribLocation(rendererProgram, "a_texcoord"),
    texture: gl.getUniformLocation(rendererProgram, "u_texture"),
    iterations: gl.getUniformLocation(rendererProgram, "iterations"),
  };

  const rendererVao = createVao(
    gl,
    [
      [
        [-1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
      ].flat(),
      [
        [-1.0, 1.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
      ].flat(),
    ],
    [rendererProgramLocations.position, rendererProgramLocations.texcoord],
    [3, 2],
    [
      [0, 1, 2],
      [1, 2, 3],
    ].flat()
  );
  if (!rendererVao) {
    console.error("Failed to create vertexArray");
    return;
  }

  const fbo = gl.createFramebuffer();
  if (!fbo) {
    console.error("Failed to create frameBuffer");
    return;
  }

  const loop = () => {
    stats.begin();

    if (value.tick || iterations < 5) {
      // render --------------------------------------------
      gl.useProgram(program);

      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, textures[0]);
      gl.uniform1i(programLocations.texture, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, triangleTexture);
      gl.uniform1i(programLocations.trianglesTexture, 1);

      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, materialTexture);
      gl.uniform1i(programLocations.materialTexture, 2);

      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, bvhTreeTexture);
      gl.uniform1i(programLocations.bvhTreeTexture, 3);

      gl.uniform1i(programLocations.iterations, iterations);
      gl.uniform2f(programLocations.resolution, canvas.width, canvas.height);
      gl.uniform3fv(programLocations.camera_position, camera.position);
      gl.uniform3fv(programLocations.camera_direction, camera.direction);
      gl.uniform3fv(programLocations.camera_up, camera.up);
      gl.uniform1f(programLocations.screen_dist, camera.screen_dist);
      gl.uniform1i(programLocations.spp, value.spp);
      gl.uniform1i(programLocations.n_triangles, triangles.length);
      gl.uniform1i(programLocations.n_materials, Object.keys(materials).length);
      gl.uniform1i(
        programLocations.render_type,
        renderTypes.indexOf(value.renderType)
      );

      gl.bindVertexArray(shaderVao);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        textures[1],
        0
      );

      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      textures.reverse();

      // renderer ------------------------------------------
      gl.useProgram(rendererProgram);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, textures[0]);
      gl.uniform1i(rendererProgramLocations.texture, 0);
      gl.uniform1i(rendererProgramLocations.iterations, iterations);

      gl.bindVertexArray(rendererVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);

      if (!value.tick) {
        diagnoseGlError(gl);
      }

      // tick ----------------------------------------------
      gl.flush();

      iterations++;
    }

    stats.end();

    requestAnimationFrame(loop);

    if (iterations % 25 == 0) {
      output.innerHTML = `iterations: ${iterations}`;
    }
  };

  requestAnimationFrame(loop);
};

main();
