import { mat4, vec3 } from "gl-matrix";

export interface Shape {
  id: string;
  type: string;
  bsdf?: {
    diffuse: boolean;
    reflectance: [number, number, number];
  };
  emitter?: {
    radiance: [number, number, number];
  };
  matrix: number[];
  model?: Triangle[];
}

export interface Scene {
  shapes: Shape[];
  sensor: {
    camera: {
      position: [number, number, number];
      direction: [number, number, number];
      up: [number, number, number];
    };
    fov: number;
  };
}

type DeepPartial<T> = { [P in keyof T]?: DeepPartial<T[P]> } | undefined;

export const loadMitsubaScene = async (
  xml: string,
  modelFiles: Record<string, () => Promise<string>>
) => {
  const parser = new DOMParser();
  const xmlDoc = parser.parseFromString(xml, "text/xml");

  const parseRgb = (element: Element): [number, number, number] => {
    const rgb = element.getAttribute("value")!.split(",");
    return [parseFloat(rgb[0]), parseFloat(rgb[1]), parseFloat(rgb[2])];
  };
  const parseBsdf = (element: Element): DeepPartial<Shape["bsdf"]> => {
    for (const child of element.children) {
      if (child.nodeName === "bsdf") {
        return {
          ...(parseBsdf(child) ?? {}),
          diffuse: true,
        };
      } else if (child.nodeName === "rgb") {
        return {
          reflectance: parseRgb(child),
        };
      }
    }
  };
  const parseShape = async (element: Element): Promise<DeepPartial<Shape>> => {
    let shape: DeepPartial<Shape> = {};

    for (const child of element.children) {
      if (child.nodeName === "bsdf") {
        shape.bsdf = parseBsdf(child);
      } else if (child.nodeName === "emitter") {
        shape.emitter = {
          radiance: parseRgb(child.children[0]),
        };
      } else if (child.nodeName === "ref") {
        shape = {
          ...shape,
          ...(await parseShape(
            xmlDoc.getElementById(child.getAttribute("id")!)!
          )),
        };
      } else if (child.nodeName === "transform") {
        shape.matrix = child.children[0]
          .getAttribute("value")!
          .split(" ")
          .map(parseFloat);
      } else if (child.nodeName === "string") {
        shape.model = await loadObjFile(
          await modelFiles[child.getAttribute("value")!]()
        );
      }
    }

    return shape;
  };

  const shapes: Shape[] = [];

  const promises: (() => Promise<void>)[] = [];
  xmlDoc.querySelectorAll("shape").forEach((shape) => {
    promises.push(async () => {
      const parsed = await parseShape(shape);

      shapes.push({
        ...parsed,
        type: shape.getAttribute("type")!,
        id: shape.getAttribute("id")!,
      } as Shape);
    });
  });

  await Promise.all(promises.map((p) => p()));

  let camera = undefined;
  let fov = undefined;

  xmlDoc
    .querySelector("sensor")!
    .querySelectorAll("matrix")
    .forEach((elem) => {
      const cam = elem.getAttribute("value")!.split(" ").map(parseFloat);
      camera = transformIntoCamera(cam);
    });
  xmlDoc
    .querySelector("sensor")!
    .querySelectorAll("float")
    .forEach((elem) => {
      fov = parseFloat(elem.getAttribute("value")!);
    });

  return {
    shapes,
    sensors: {
      camera: camera!,
      fov: fov!,
    },
  };
};

export const transformIntoCamera = (matrix: number[]) => {
  return {
    position: [matrix[3], matrix[7], matrix[11]] as [number, number, number],
    direction: [matrix[2], matrix[6], matrix[10]] as [number, number, number],
    up: [matrix[1], matrix[5], matrix[9]] as [number, number, number],
  };
};

export const loadMat4 = (matrix: number[]) => {
  return mat4.fromValues(
    matrix[0],
    matrix[4],
    matrix[8],
    matrix[12],
    matrix[1],
    matrix[5],
    matrix[9],
    matrix[13],
    matrix[2],
    matrix[6],
    matrix[10],
    matrix[14],
    matrix[3],
    matrix[7],
    matrix[11],
    matrix[15]
  );
};

export interface Triangle {
  vertices: [vec3, vec3, vec3];
  normals: [vec3, vec3, vec3];
}

export const loadObjFile = async (raw: string) => {
  const vertices: vec3[] = [];
  const normals: vec3[] = [];
  const faces: Triangle[] = [];
  raw.split("\n").forEach((line) => {
    if (line.startsWith("v ")) {
      const [x, y, z] = line.split(" ").slice(1).map(parseFloat);
      vertices.push([x, y, z]);
    } else if (line.startsWith("vn ")) {
      const [x, y, z] = line.split(" ").slice(1).map(parseFloat);
      normals.push([x, y, z]);
    } else if (line.startsWith("f ")) {
      const [t1, t2, t3] = line
        .split(" ")
        .slice(1)
        .map((v) => {
          const [vi, ti, ni] = v.split("/").map((i) => parseInt(i) - 1);
          return [vi, ti, ni] as [number, number, number];
        });
      faces.push({
        vertices: [vertices[t1[0]], vertices[t2[0]], vertices[t3[0]]],
        normals: [normals[t1[2]], normals[t2[2]], normals[t3[2]]],
      });
    }
  });

  return faces;
};

const tokenizeMtl = (
  raw: string
): (
  | {
      type: "identifier";
      value: string;
    }
  | {
      type: "number";
      value: number;
    }
  | {
      type: "keyword";
      value: string;
    }
)[] => {
  let position = 0;
  const tokens: (
    | {
        type: "identifier";
        value: string;
      }
    | {
        type: "number";
        value: number;
      }
    | {
        type: "keyword";
        value: string;
      }
  )[] = [];

  while (position < raw.length) {
    if (raw[position] === "#") {
      while (raw[position] !== "\n") {
        position++;
      }

      continue;
    }

    if (raw[position].match(/\s/)) {
      position++;
      continue;
    }

    const keywords = [
      "newmtl",
      "Ns",
      "Ni",
      "illum",
      "Ka",
      "Kd",
      "Ks",
      "Ke",
      "d",
      "Tr",
      "Tf",
    ];
    let should_continue = false;
    for (const keyword of keywords) {
      if (raw.slice(position).startsWith(keyword)) {
        tokens.push({
          type: "keyword",
          value: keyword,
        });
        position += keyword.length;
        should_continue = true;
        break;
      }
    }
    if (should_continue) continue;

    if (raw[position].match(/[a-zA-Z]/)) {
      const start = position;
      while (raw[position].match(/[a-zA-Z]/)) {
        position++;
      }
      tokens.push({
        type: "identifier",
        value: raw.slice(start, position),
      });
      continue;
    }
    if (raw[position].match(/[0-9\.]/)) {
      const start = position;
      while (position < raw.length && raw[position].match(/[0-9\.]/)) {
        position++;
      }
      tokens.push({
        type: "number",
        value: parseFloat(raw.slice(start, position)),
      });
      continue;
    }

    throw new Error(
      `Unexpected token: ${raw[position].charCodeAt(0)}, ${raw.slice(
        position - 20,
        position
      )}\n@@@\n${raw.slice(position, position + 50)}`
    );
  }

  return tokens;
};

export interface Material {
  name: string;
  Ns?: number;
  Ni?: number;
  illum?: number;
  Ka?: vec3;
  Kd?: vec3;
  Ks?: vec3;
  Ke?: vec3;
  d?: number;
  Tr?: number;
  Tf?: vec3;
}

export const loadMtlScene = (raw: string) => {
  const tokens = tokenizeMtl(raw);
  let position = 0;

  const materials: Material[] = [];

  while (position < tokens.length) {
    const token = tokens[position];

    if (token.type === "keyword" && token.value === "newmtl") {
      const nextToken = tokens[position + 1];
      if (nextToken.type !== "identifier") {
        throw new Error(`Unexpected token: ${nextToken}`);
      }

      const name = nextToken.value;
      const material = {
        name,
      };
      materials.push(material);
      position += 2;
      continue;
    } else if (token.type === "keyword" && token.value === "Ns") {
      const nextToken = tokens[position + 1];
      if (nextToken.type !== "number") {
        throw new Error(`Unexpected token: ${nextToken}`);
      }
      materials[materials.length - 1].Ns = nextToken.value;
      position += 2;
      continue;
    } else if (token.type === "keyword" && token.value === "Ni") {
      const nextToken = tokens[position + 1];
      if (nextToken.type !== "number") {
        throw new Error(`Unexpected token: ${nextToken}`);
      }
      materials[materials.length - 1].Ni = nextToken.value;
      position += 2;
      continue;
    } else if (token.type === "keyword" && token.value === "illum") {
      const nextToken = tokens[position + 1];
      if (nextToken.type !== "number") {
        throw new Error(`Unexpected token: ${nextToken}`);
      }
      materials[materials.length - 1].illum = nextToken.value;
      position += 2;
      continue;
    } else if (token.type === "keyword" && token.value === "Ka") {
      const r = tokens[position + 1];
      const g = tokens[position + 2];
      const b = tokens[position + 3];

      if (r.type !== "number" || g.type !== "number" || b.type !== "number") {
        throw new Error(`Unexpected token: ${r} ${g} ${b}`);
      }

      materials[materials.length - 1].Ka = [r.value, g.value, b.value];
      position += 4;
      continue;
    } else if (token.type === "keyword" && token.value === "Kd") {
      const r = tokens[position + 1];
      const g = tokens[position + 2];
      const b = tokens[position + 3];

      if (r.type !== "number" || g.type !== "number" || b.type !== "number") {
        throw new Error(`Unexpected token: ${r} ${g} ${b}`);
      }

      materials[materials.length - 1].Kd = [r.value, g.value, b.value];
      position += 4;
      continue;
    } else if (token.type === "keyword" && token.value === "Ks") {
      const r = tokens[position + 1];
      const g = tokens[position + 2];
      const b = tokens[position + 3];

      if (r.type !== "number" || g.type !== "number" || b.type !== "number") {
        throw new Error(`Unexpected token: ${r} ${g} ${b}`);
      }

      materials[materials.length - 1].Ks = [r.value, g.value, b.value];
      position += 4;
      continue;
    } else if (token.type === "keyword" && token.value === "Ke") {
      const r = tokens[position + 1];
      const g = tokens[position + 2];
      const b = tokens[position + 3];

      if (r.type !== "number" || g.type !== "number" || b.type !== "number") {
        throw new Error(`Unexpected token: ${r} ${g} ${b}`);
      }

      materials[materials.length - 1].Ke = [r.value, g.value, b.value];
      position += 4;
      continue;
    } else if (token.type === "keyword" && token.value === "d") {
      const nextToken = tokens[position + 1];
      if (nextToken.type !== "number") {
        throw new Error(`Unexpected token: ${nextToken}`);
      }
      materials[materials.length - 1].d = nextToken.value;
      position += 2;
      continue;
    } else if (token.type === "keyword" && token.value === "Tr") {
      const nextToken = tokens[position + 1];
      if (nextToken.type !== "number") {
        throw new Error(`Unexpected token: ${nextToken}`);
      }
      materials[materials.length - 1].Tr = nextToken.value;
      position += 2;
      continue;
    } else if (token.type === "keyword" && token.value === "Tf") {
      const r = tokens[position + 1];
      const g = tokens[position + 2];
      const b = tokens[position + 3];

      if (r.type !== "number" || g.type !== "number" || b.type !== "number") {
        throw new Error(`Unexpected token: ${r} ${g} ${b}`);
      }

      materials[materials.length - 1].Tf = [r.value, g.value, b.value];
      position += 4;
      continue;
    } else {
      throw new Error(`Unexpected token: ${token}`);
    }
  }

  return materials;
};

const tokenizeObj = (raw: string) => {
  let position = 0;
  const tokens: (
    | {
        type: "identifier";
        value: string;
      }
    | {
        type: "number";
        value: number;
      }
    | {
        type: "keyword";
        value: string;
      }
    | {
        type: "slash";
      }
  )[] = [];

  while (position < raw.length) {
    if (raw[position] === "#") {
      while (raw[position] !== "\n") {
        position++;
      }

      continue;
    }

    if (raw[position].match(/\s/)) {
      position++;
      continue;
    }

    const keywords = ["mtllib", "vt", "vn", "v", "g", "usemtl", "f", "s"];
    let should_continue = false;
    for (const keyword of keywords) {
      if (
        raw.slice(position).startsWith(keyword) &&
        raw[position + keyword.length].match(/\s/)
      ) {
        tokens.push({
          type: "keyword",
          value: keyword,
        });
        position += keyword.length;
        should_continue = true;
        break;
      }
    }
    if (should_continue) continue;

    if (raw.slice(position, position + 2).match(/[a-zA-Z][a-zA-Z\-\.]/)) {
      const start = position;
      while (position < raw.length && raw[position].match(/[a-zA-Z\-\.]/)) {
        position++;
      }
      tokens.push({
        type: "identifier",
        value: raw.slice(start, position),
      });
      continue;
    }
    if (raw[position].match(/[0-9\.\-]/)) {
      const start = position;
      while (position < raw.length && raw[position].match(/[0-9\.\-]/)) {
        position++;
      }
      tokens.push({
        type: "number",
        value: parseFloat(raw.slice(start, position)),
      });
      continue;
    }
    if (raw[position] === "/") {
      position++;
      tokens.push({
        type: "slash",
      });
      continue;
    }

    console.log(raw[position].charCodeAt(0));

    throw new Error(`Unexpected token: ${raw.slice(position, position + 50)}`);
  }

  return tokens;
};

export interface SceneObj {
  mtllib?: string;
  objects: {
    name: string;
    faces: {
      vertices: vec3[];
      normals: vec3[];
    }[];
    usemtl: string;
    smooth?: boolean;
  }[];
}

export const loadObjScene = (raw: string) => {
  let position = 0;
  const tokens = tokenizeObj(raw);

  const scene: SceneObj = {
    objects: [],
  };
  let vertices: vec3[] = [];
  let vt: vec3[] = [];
  let vn: vec3[] = [];

  while (position < tokens.length) {
    const token = tokens[position];
    if (token.type === "keyword" && token.value === "mtllib") {
      const nextToken = tokens[position + 1];
      if (nextToken.type !== "identifier") {
        throw new Error(`Unexpected token: ${nextToken}`);
      }
      scene.mtllib = nextToken.value;
      position += 2;
      continue;
    } else if (token.type === "keyword" && token.value === "v") {
      const x = tokens[position + 1];
      const y = tokens[position + 2];
      const z = tokens[position + 3];

      if (x.type !== "number" || y.type !== "number" || z.type !== "number") {
        throw new Error(
          `Unexpected token [v]: ${JSON.stringify(x)} ${JSON.stringify(
            y
          )} ${JSON.stringify(z)}`
        );
      }

      vertices.push([x.value, y.value, z.value]);
      position += 4;
      continue;
    } else if (token.type === "keyword" && token.value === "vt") {
      const x = tokens[position + 1];
      const y = tokens[position + 2];
      const z = tokens[position + 3];

      if (x.type !== "number" || y.type !== "number" || z.type !== "number") {
        throw new Error(
          `Unexpected token [vt]: ${JSON.stringify(x)} ${JSON.stringify(
            y
          )} ${JSON.stringify(z)}`
        );
      }

      vt.push([x.value, y.value, z.value]);
      position += 4;
      continue;
    } else if (token.type === "keyword" && token.value === "vn") {
      const x = tokens[position + 1];
      const y = tokens[position + 2];
      const z = tokens[position + 3];

      if (x.type !== "number" || y.type !== "number" || z.type !== "number") {
        throw new Error(
          `Unexpected token [vn]: ${JSON.stringify(x)} ${JSON.stringify(
            y
          )} ${JSON.stringify(z)}`
        );
      }

      vn.push([x.value, y.value, z.value]);
      position += 4;
      continue;
    } else if (token.type === "keyword" && token.value === "g") {
      const nextToken = tokens[position + 1];
      if (nextToken.type !== "identifier") {
        throw new Error(`Unexpected token [g]: ${JSON.stringify(nextToken)}`);
      }
      const name = nextToken.value;

      if (scene.objects[scene.objects!.length - 1]?.name === name) {
        position += 2;
        continue;
      } else {
        const object = {
          name,
          faces: [],
          usemtl: "",
        };
        scene.objects.push(object);
        position += 2;
        continue;
      }
    } else if (token.type === "keyword" && token.value === "usemtl") {
      const nextToken = tokens[position + 1];
      if (nextToken.type !== "identifier") {
        throw new Error(`Unexpected token: ${nextToken}`);
      }

      if (scene.objects[scene.objects!.length - 1].usemtl === "") {
        scene.objects[scene.objects!.length - 1].usemtl = nextToken.value;
      } else {
        const name = nextToken.value;
        const object = {
          name,
          faces: [],
          usemtl: "",
        };
        scene.objects.push(object);
      }
      position += 2;
      continue;
    } else if (token.type === "keyword" && token.value === "f") {
      const parseV = () => {
        const v = tokens[position];
        if (v.type !== "number") {
          throw new Error(`Unexpected token [f]: ${JSON.stringify(v)}`);
        }
        position++;

        if (position >= tokens.length) {
          return [v.value];
        }

        if (tokens[position].type === "slash") {
          position++;
          const vt = tokens[position];
          if (vt.type !== "number") {
            throw new Error(`Unexpected token [f]: ${vt}`);
          }
          position++;

          if (tokens[position].type === "slash") {
            position++;
            const vn = tokens[position];
            if (vn.type !== "number") {
              throw new Error(`Unexpected token [f]: ${vn}`);
            }
            position++;

            return [v.value, vt.value, vn.value];
          } else {
            return [v.value, vt.value];
          }
        } else {
          return [v.value];
        }
      };

      position++;
      const v1 = parseV();
      const v2 = parseV();
      const v3 = parseV();
      let v4 = undefined;
      const v4Token = tokens[position];
      if (position < tokens.length && v4Token.type === "number") {
        v4 = parseV();
      }

      scene.objects[scene.objects!.length - 1].faces.push({
        vertices: [
          vertices[v1[0] < 0 ? vertices.length + v1[0] : v1[0]],
          vertices[v2[0] < 0 ? vertices.length + v2[0] : v2[0]],
          vertices[v3[0] < 0 ? vertices.length + v3[0] : v3[0]],
          v4
            ? vertices[v4[0] < 0 ? vertices.length + v4[0] : v4[0]]
            : undefined,
        ].filter((v: vec3 | undefined): v is vec3 => !!v),
        normals:
          v1[2] && v2[2] && v3[2]
            ? [
                vn[v1[2] < 0 ? vn.length + v1[2] : v1[2]],
                vn[v2[2] < 0 ? vn.length + v2[2] : v2[2]],
                vn[v3[2] < 0 ? vn.length + v3[2] : v3[2]],
                v4 ? vn[v4[2] < 0 ? vn.length + v4[2] : v4[2]] : undefined,
              ].filter((v: vec3 | undefined): v is vec3 => !!v)
            : [],
      });
      continue;
    } else if (token.type === "keyword" && token.value === "s") {
      position++;
      const value = tokens[position];
      if (value.type !== "number") {
        throw new Error(`Unexpected token: ${JSON.stringify(value)}`);
      }
      scene.objects[scene.objects!.length - 1].smooth = value.value === 1;

      position++;
      continue;
    } else {
      throw new Error(`Unexpected token: ${JSON.stringify(token)}`);
    }
  }

  return scene;
};
