/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : common_def.h
 * Authors     : lzwang
 * Create Time : 2022-05-23 11:30:24 (CST)
 * Description :
 *
 */

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_DEF_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_DEF_H__

#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_DLL
#ifdef __GNUC__
#define VENUS_EXPORTS __attribute__((dllexport))
#else
#define VENUS_EXPORTS __declspec(dllexport)
#endif
#else
#ifdef __GNUC__
#define VENUS_EXPORTS __attribute__((dllimport))
#else
#define VENUS_EXPORTS __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define VENUS_EXPORTS __attribute__((visibility("default")))
#else
#define VENUS_EXPORTS
#endif
#endif

#ifndef VENUS_EXTERN_C
#ifdef __cplusplus
#define VENUS_EXTERN_C extern "C"
#else
#define VENUS_EXTERN_C
#endif
#endif
#include <stdint.h>

#define VENUS_API VENUS_EXPORTS
#define VENUS_C_API VENUS_EXTERN_C VENUS_EXPORTS

#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_DEF_H__ */
