#pragma once
#include <SDL.h>
#include "glad.h"
#include <iostream>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <assimp/scene.h>

typedef unsigned int uint;
typedef unsigned char byte;

inline glm::mat4 assimpToGlmMatrix(aiMatrix4x4 mat) {
	glm::mat4 m;
	for (int y = 0; y < 4; y++)
	{
		for (int x = 0; x < 4; x++)
		{
			m[x][y] = mat[y][x];
		}
	}
	return m;
}
inline glm::vec3 assimpToGlmVec3(aiVector3D vec) {
	return glm::vec3(vec.x, vec.y, vec.z);
}

inline glm::quat assimpToGlmQuat(aiQuaternion quat) {
	glm::quat q;
	q.x = quat.x;
	q.y = quat.y;
	q.z = quat.z;
	q.w = quat.w;

	return q;
}



inline unsigned int createShader(const char* vertexStr, const char* fragmentStr) {
	int success;
	char info_log[512];
	uint
		program = glCreateProgram(),
		vShader = glCreateShader(GL_VERTEX_SHADER),
		fShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vShader, 1, &vertexStr, 0);
	glCompileShader(vShader);
	glGetShaderiv(vShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vShader, 512, 0, info_log);
		std::cout << "vertex shader compilation failed!\n" << info_log << std::endl;
	}
	glShaderSource(fShader, 1, &fragmentStr, 0);
	glCompileShader(fShader);
	glGetShaderiv(fShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fShader, 512, 0, info_log);
		std::cout << "fragment shader compilation failed!\n" << info_log << std::endl;
	}

	glAttachShader(program, vShader);
	glAttachShader(program, fShader);
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(program, 512, 0, info_log);
		std::cout << "program linking failed!\n" << info_log << std::endl;
	}
	glDetachShader(program, vShader);
	glDeleteShader(vShader);
	glDetachShader(program, fShader);
	glDeleteShader(fShader);

	return program;
}

inline SDL_Window* initWindow(int &windowWidth,int &windowHeight) {
	SDL_Init(SDL_INIT_EVERYTHING);
	SDL_GL_LoadLibrary(NULL);

	//window
	SDL_Window* window = SDL_CreateWindow("skin Animation",
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		640, 480,
		SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);


	SDL_GLContext context = SDL_GL_CreateContext(window);

	SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);


	gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress);

	SDL_GetWindowSize(window, &windowWidth, &windowHeight);
	glViewport(0, 0, windowWidth, windowHeight);
	glClearColor(1.0, 0.0, 0.4, 1.0);
	glEnable(GL_DEPTH_TEST);
	SDL_ShowWindow(window);
	SDL_GL_SetSwapInterval(1);
	return window;
}