/*
 * MVKDescriptor.mm
 *
 * Copyright (c) 2015-2021 The Brenwill Workshop Ltd. (http://www.brenwill.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MVKDescriptor.h"
#include "MVKDescriptorSet.h"
#include "MVKCommandBuffer.h"
#include "MVKBuffer.h"

using namespace SPIRV_CROSS_NAMESPACE;


#pragma mark MVKShaderStageResourceBinding

MVKShaderStageResourceBinding MVKShaderStageResourceBinding::operator+ (const MVKShaderStageResourceBinding& rhs) {
	MVKShaderStageResourceBinding rslt;
	rslt.resourceIndex = this->resourceIndex + rhs.resourceIndex;
	rslt.bufferIndex = this->bufferIndex + rhs.bufferIndex;
	rslt.textureIndex = this->textureIndex + rhs.textureIndex;
	rslt.samplerIndex = this->samplerIndex + rhs.samplerIndex;
	return rslt;
}

MVKShaderStageResourceBinding& MVKShaderStageResourceBinding::operator+= (const MVKShaderStageResourceBinding& rhs) {
	this->resourceIndex += rhs.resourceIndex;
	this->bufferIndex += rhs.bufferIndex;
	this->textureIndex += rhs.textureIndex;
	this->samplerIndex += rhs.samplerIndex;
	return *this;
}

void MVKShaderStageResourceBinding::addArgumentBuffer(const MVKShaderStageResourceBinding& rhs) {
	bool isUsed = rhs.resourceIndex > 0;
	this->bufferIndex += isUsed;
	this->resourceIndex += isUsed;
}


#pragma mark MVKShaderResourceBinding

uint16_t MVKShaderResourceBinding::getMaxBufferIndex() {
	return std::max({stages[kMVKShaderStageVertex].bufferIndex, stages[kMVKShaderStageTessCtl].bufferIndex, stages[kMVKShaderStageTessEval].bufferIndex, stages[kMVKShaderStageFragment].bufferIndex, stages[kMVKShaderStageCompute].bufferIndex});
}

uint16_t MVKShaderResourceBinding::getMaxTextureIndex() {
	return std::max({stages[kMVKShaderStageVertex].textureIndex, stages[kMVKShaderStageTessCtl].textureIndex, stages[kMVKShaderStageTessEval].textureIndex, stages[kMVKShaderStageFragment].textureIndex, stages[kMVKShaderStageCompute].textureIndex});
}

uint16_t MVKShaderResourceBinding::getMaxSamplerIndex() {
	return std::max({stages[kMVKShaderStageVertex].samplerIndex, stages[kMVKShaderStageTessCtl].samplerIndex, stages[kMVKShaderStageTessEval].samplerIndex, stages[kMVKShaderStageFragment].samplerIndex, stages[kMVKShaderStageCompute].samplerIndex});
}

MVKShaderResourceBinding MVKShaderResourceBinding::operator+ (const MVKShaderResourceBinding& rhs) {
	MVKShaderResourceBinding rslt;
	for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
		rslt.stages[i] = this->stages[i] + rhs.stages[i];
	}
	return rslt;
}

MVKShaderResourceBinding& MVKShaderResourceBinding::operator+= (const MVKShaderResourceBinding& rhs) {
	for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
		this->stages[i] += rhs.stages[i];
	}
	return *this;
}

void MVKShaderResourceBinding::addArgumentBuffer(const MVKShaderResourceBinding& rhs) {
	for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
		this->stages[i].addArgumentBuffer(rhs.stages[i]);
	}
}

void mvkPopulateShaderConverterContext(mvk::SPIRVToMSLConversionConfiguration& shaderConfig,
									   MVKShaderStageResourceBinding& ssRB,
									   spv::ExecutionModel stage,
									   uint32_t descriptorSetIndex,
									   uint32_t bindingIndex,
									   uint32_t count,
									   MVKSampler* immutableSampler) {
	mvk::MSLResourceBinding rb;

	auto& rbb = rb.resourceBinding;
	rbb.stage = stage;
	rbb.desc_set = descriptorSetIndex;
	rbb.binding = bindingIndex;
	rbb.count = count;
	rbb.msl_buffer = ssRB.bufferIndex;
	rbb.msl_texture = ssRB.textureIndex;
	rbb.msl_sampler = ssRB.samplerIndex;

	if (immutableSampler) { immutableSampler->getConstexprSampler(rb); }

	shaderConfig.resourceBindings.push_back(rb);
}

MTLRenderStages mvkMTLRenderStagesFromMVKShaderStages(bool stageEnabled[]) {
	MTLRenderStages mtlStages = 0;
	for (uint32_t stage = kMVKShaderStageVertex; stage < kMVKShaderStageCount; stage++) {
		if (stageEnabled[stage]) {
			switch (stage) {
				case kMVKShaderStageVertex:
				case kMVKShaderStageTessCtl:
				case kMVKShaderStageTessEval:
					mtlStages |= MTLRenderStageVertex;
					break;

				case kMVKShaderStageFragment:
					mtlStages |= MTLRenderStageFragment;
					break;

				default:
					break;
			}
		}
	}
	return mtlStages;
}


#pragma mark -
#pragma mark MVKDescriptorSetLayoutBinding

MVKVulkanAPIObject* MVKDescriptorSetLayoutBinding::getVulkanAPIObject() { return _layout; };

uint32_t MVKDescriptorSetLayoutBinding::getDescriptorCount(MVKDescriptorSet* descSet) {

	if (_info.descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT) {
		return 1;
	}

	if (descSet && mvkIsAnyFlagEnabled(_flags, VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT)) {
		return descSet->_variableDescriptorCount;
	}

	return _info.descriptorCount;
}

MVKSampler* MVKDescriptorSetLayoutBinding::getImmutableSampler(uint32_t index) {
	return (index < _immutableSamplers.size()) ? _immutableSamplers[index] : nullptr;
}

// A null cmdEncoder can be passed to perform a validation pass
void MVKDescriptorSetLayoutBinding::bind(MVKCommandEncoder* cmdEncoder,
										 uint32_t descSetIndex,
										 MVKDescriptorSet* descSet,
										 MVKShaderResourceBinding& dslMTLRezIdxOffsets,
										 MVKArrayRef<uint32_t> dynamicOffsets,
										 uint32_t& dynamicOffsetIndex) {

	// Establish the resource indices to use, by combining the offsets of the DSL and this DSL binding.
    MVKShaderResourceBinding mtlIdxs = _mtlResourceIndexOffsets + dslMTLRezIdxOffsets;

	VkDescriptorType descType = getDescriptorType();
    uint32_t descCnt = getDescriptorCount(descSet);
    for (uint32_t descIdx = 0; descIdx < descCnt; descIdx++) {
		MVKDescriptor* mvkDesc = descSet->getDescriptor(getBinding(), descIdx);
		if (mvkDesc->getDescriptorType() == descType) {
			mvkDesc->bind(cmdEncoder, descSetIndex, descIdx, this, _applyToStage,
						  mtlIdxs, dynamicOffsets, dynamicOffsetIndex);
		}
    }
}

template<typename T>
static const T& get(const void* pData, size_t stride, uint32_t index) {
    return *(T*)((const char*)pData + stride * index);
}

// A null cmdEncoder can be passed to perform a validation pass
void MVKDescriptorSetLayoutBinding::push(MVKCommandEncoder* cmdEncoder,
                                         uint32_t& dstArrayElement,
                                         uint32_t& descriptorCount,
                                         uint32_t& descriptorsPushed,
                                         VkDescriptorType descriptorType,
                                         size_t stride,
                                         const void* pData,
                                         MVKShaderResourceBinding& dslMTLRezIdxOffsets) {
    MVKMTLBufferBinding bb;
    MVKMTLTextureBinding tb;
    MVKMTLSamplerStateBinding sb;

    if (dstArrayElement >= _info.descriptorCount) {
        dstArrayElement -= _info.descriptorCount;
        return;
    }

    if (descriptorType != _info.descriptorType) {
        dstArrayElement = 0;
        if (_info.descriptorCount > descriptorCount)
            descriptorCount = 0;
        else {
            descriptorCount -= _info.descriptorCount;
            descriptorsPushed = _info.descriptorCount;
        }
        return;
    }

    // Establish the resource indices to use, by combining the offsets of the DSL and this DSL binding.
    MVKShaderResourceBinding mtlIdxs = _mtlResourceIndexOffsets + dslMTLRezIdxOffsets;

    for (uint32_t rezIdx = dstArrayElement;
         rezIdx < _info.descriptorCount && rezIdx - dstArrayElement < descriptorCount;
         rezIdx++) {
        switch (_info.descriptorType) {

            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
                const auto& bufferInfo = get<VkDescriptorBufferInfo>(pData, stride, rezIdx - dstArrayElement);
                MVKBuffer* buffer = (MVKBuffer*)bufferInfo.buffer;
                bb.mtlBuffer = buffer->getMTLBuffer();
                bb.offset = buffer->getMTLBufferOffset() + bufferInfo.offset;
                if (bufferInfo.range == VK_WHOLE_SIZE)
                    bb.size = (uint32_t)(buffer->getByteCount() - bb.offset);
                else
                    bb.size = (uint32_t)bufferInfo.range;

                for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
                    if (_applyToStage[i]) {
                        bb.index = mtlIdxs.stages[i].bufferIndex + rezIdx;
						if (cmdEncoder) { cmdEncoder->bindBuffer(bb, MVKShaderStage(i)); }
                    }
                }
                break;
            }

            case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT: {
                const auto& inlineUniformBlock = *(VkWriteDescriptorSetInlineUniformBlockEXT*)pData;
                bb.mtlBytes = inlineUniformBlock.pData;
                bb.size = inlineUniformBlock.dataSize;
                bb.isInline = true;
                for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
                    if (_applyToStage[i]) {
                        bb.index = mtlIdxs.stages[i].bufferIndex;
						if (cmdEncoder) { cmdEncoder->bindBuffer(bb, MVKShaderStage(i)); }
                    }
                }
                break;
            }

            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
                const auto& imageInfo = get<VkDescriptorImageInfo>(pData, stride, rezIdx - dstArrayElement);
                MVKImageView* imageView = (MVKImageView*)imageInfo.imageView;
                uint8_t planeCount = (imageView) ? imageView->getPlaneCount() : 1;
                for (uint8_t planeIndex = 0; planeIndex < planeCount; planeIndex++) {
                    tb.mtlTexture = imageView->getMTLTexture(planeIndex);
                    tb.swizzle = (_info.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) ? imageView->getPackedSwizzle() : 0;
                    if (_info.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                        id<MTLTexture> mtlTex = tb.mtlTexture;
                        if (mtlTex.parentTexture) { mtlTex = mtlTex.parentTexture; }
                        bb.mtlBuffer = mtlTex.buffer;
                        bb.offset = mtlTex.bufferOffset;
                        bb.size = (uint32_t)(mtlTex.height * mtlTex.bufferBytesPerRow);
                    }
                    for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
                        if (_applyToStage[i]) {
                            tb.index = mtlIdxs.stages[i].textureIndex + rezIdx + planeIndex;
							if (cmdEncoder) { cmdEncoder->bindTexture(tb, MVKShaderStage(i)); }
                            if (_info.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                                bb.index = mtlIdxs.stages[i].bufferIndex + rezIdx;
								if (cmdEncoder) { cmdEncoder->bindBuffer(bb, MVKShaderStage(i)); }
                            }
                        }
                    }
                }
                break;
            }

            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
                auto* bufferView = get<MVKBufferView*>(pData, stride, rezIdx - dstArrayElement);
                tb.mtlTexture = bufferView->getMTLTexture();
                tb.swizzle = 0;
                if (_info.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                    id<MTLTexture> mtlTex = tb.mtlTexture;
                    bb.mtlBuffer = mtlTex.buffer;
                    bb.offset = mtlTex.bufferOffset;
                    bb.size = (uint32_t)(mtlTex.height * mtlTex.bufferBytesPerRow);
                }
                for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
                    if (_applyToStage[i]) {
                        tb.index = mtlIdxs.stages[i].textureIndex + rezIdx;
						if (cmdEncoder) { cmdEncoder->bindTexture(tb, MVKShaderStage(i)); }
                        if (_info.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                            bb.index = mtlIdxs.stages[i].bufferIndex + rezIdx;
							if (cmdEncoder) { cmdEncoder->bindBuffer(bb, MVKShaderStage(i)); }
                        }
                    }
                }
                break;
            }

            case VK_DESCRIPTOR_TYPE_SAMPLER: {
                MVKSampler* sampler;
				if (_immutableSamplers.empty()) {
                    sampler = (MVKSampler*)get<VkDescriptorImageInfo>(pData, stride, rezIdx - dstArrayElement).sampler;
					validate(sampler);
				} else {
                    sampler = _immutableSamplers[rezIdx];
				}
                sb.mtlSamplerState = sampler->getMTLSamplerState();
                for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
                    if (_applyToStage[i]) {
                        sb.index = mtlIdxs.stages[i].samplerIndex + rezIdx;
						if (cmdEncoder) { cmdEncoder->bindSamplerState(sb, MVKShaderStage(i)); }
                    }
                }
                break;
            }

            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
                const auto& imageInfo = get<VkDescriptorImageInfo>(pData, stride, rezIdx - dstArrayElement);
                MVKImageView* imageView = (MVKImageView*)imageInfo.imageView;
                uint8_t planeCount = (imageView) ? imageView->getPlaneCount() : 1;
                for (uint8_t planeIndex = 0; planeIndex < planeCount; planeIndex++) {
                    tb.mtlTexture = imageView->getMTLTexture(planeIndex);
                    tb.swizzle = (imageView) ? imageView->getPackedSwizzle() : 0;
                    MVKSampler* sampler;
                    if (_immutableSamplers.empty()) {
                        sampler = (MVKSampler*)imageInfo.sampler;
                        validate(sampler);
                    } else {
                        sampler = _immutableSamplers[rezIdx];
                    }
                    sb.mtlSamplerState = sampler->getMTLSamplerState();
                    for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
                        if (_applyToStage[i]) {
                            tb.index = mtlIdxs.stages[i].textureIndex + rezIdx + planeIndex;
                            sb.index = mtlIdxs.stages[i].samplerIndex + rezIdx;
							if (cmdEncoder) {
								cmdEncoder->bindTexture(tb, MVKShaderStage(i));
								cmdEncoder->bindSamplerState(sb, MVKShaderStage(i));
							}
                        }
                    }
                }
                break;
            }

            default:
                break;
        }
    }

    dstArrayElement = 0;
    if (_info.descriptorCount > descriptorCount)
        descriptorCount = 0;
    else {
        descriptorCount -= _info.descriptorCount;
        descriptorsPushed = _info.descriptorCount;
    }
}

bool MVKDescriptorSetLayoutBinding::isUsingMetalArgumentBuffer() const  { return _layout->isUsingMetalArgumentBuffer(); };

// Inits the index into the Metal argument buffer for this binding, and updates resource indexes consumed.
void MVKDescriptorSetLayoutBinding::initMetalArgumentBufferIndexes(uint32_t& argIdx, NSUInteger& argBuffSize) {

	_argumentBufferIndex = argIdx;

	uint32_t descCnt = getDescriptorCount();
	switch (getDescriptorType()) {

		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
			argIdx += descCnt;
			argBuffSize += sizeof(id<MTLBuffer>) * descCnt;
			break;

		case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
			argIdx += 1;
			argBuffSize += (MVKInlineUniformBlockDescriptor::shouldEmbedInlineBlocksInMetalAgumentBuffer()
							? _info.descriptorCount
							: sizeof(id<MTLBuffer>));
			break;

		case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
		case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
		case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
			argIdx += descCnt;
			argBuffSize += sizeof(id<MTLTexture>) * descCnt;
			break;

		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
		case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
			argIdx += descCnt;
			argBuffSize += sizeof(id<MTLTexture>) * descCnt;		// Texture
			argIdx += descCnt;
			argBuffSize += sizeof(id<MTLBuffer>) * descCnt;			// Buffer for atomic operations
			break;

		case VK_DESCRIPTOR_TYPE_SAMPLER:
			argIdx += descCnt;
			argBuffSize += sizeof(id<MTLSamplerState>) * descCnt;
			break;

		case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
			argIdx += descCnt;										// Texture
			argBuffSize += sizeof(id<MTLTexture>) * descCnt;
			argIdx += descCnt;										// Sampler
			argBuffSize += sizeof(id<MTLSamplerState>) * descCnt;
			break;

		default:
			break;
	}
}

// Adds MTLArgumentDescriptors to the array, and updates resource indexes consumed.
void MVKDescriptorSetLayoutBinding::addMTLArgumentDescriptors(NSMutableArray<MTLArgumentDescriptor*>* args,
															  mvk::SPIRVToMSLConversionConfiguration& shaderConfig,
															  uint32_t descSetIdx) {
	switch (getDescriptorType()) {

		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
			addMTLArgumentDescriptor(args, MTLDataTypePointer, MTLArgumentAccessReadOnly, shaderConfig, descSetIdx);
			break;

		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
			addMTLArgumentDescriptor(args, MTLDataTypePointer, MTLArgumentAccessReadWrite, shaderConfig, descSetIdx);
			break;

		case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
			if (MVKInlineUniformBlockDescriptor::shouldEmbedInlineBlocksInMetalAgumentBuffer()) {
				addMTLArgumentDescriptor(args, MTLDataTypeUChar, MTLArgumentAccessReadOnly, shaderConfig, descSetIdx);
			} else {
				addMTLArgumentDescriptor(args, MTLDataTypePointer, MTLArgumentAccessReadOnly, shaderConfig, descSetIdx);
			}
			break;

		case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
		case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
			addMTLArgumentDescriptor(args, MTLDataTypeTexture, MTLArgumentAccessReadOnly, shaderConfig, descSetIdx);
			break;

		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
			addMTLArgumentDescriptor(args, MTLDataTypeTexture, MTLArgumentAccessReadWrite, shaderConfig, descSetIdx);
			addMTLArgumentDescriptor(args, MTLDataTypePointer, MTLArgumentAccessReadWrite, shaderConfig, descSetIdx, getDescriptorCount());		// Needed for atomic operations
			break;

		case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
			addMTLArgumentDescriptor(args, MTLDataTypeTexture, MTLArgumentAccessReadOnly, shaderConfig, descSetIdx);
			break;

		case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
			addMTLArgumentDescriptor(args, MTLDataTypeTexture, MTLArgumentAccessReadWrite, shaderConfig, descSetIdx);
			addMTLArgumentDescriptor(args, MTLDataTypePointer, MTLArgumentAccessReadWrite, shaderConfig, descSetIdx, getDescriptorCount());		// Needed for atomic operations
			break;

		case VK_DESCRIPTOR_TYPE_SAMPLER:
			addMTLArgumentDescriptor(args, MTLDataTypeSampler, MTLArgumentAccessReadOnly, shaderConfig, descSetIdx);
			break;

		case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
			addMTLArgumentDescriptor(args, MTLDataTypeTexture, MTLArgumentAccessReadOnly, shaderConfig, descSetIdx);
			addMTLArgumentDescriptor(args, MTLDataTypeSampler, MTLArgumentAccessReadOnly, shaderConfig, descSetIdx, getDescriptorCount());
			break;

		default:
			break;
	}
}

// Adds an MTLArgumentDescriptor if the specified type to the array, and updates resource indexes consumed.
void MVKDescriptorSetLayoutBinding::addMTLArgumentDescriptor(NSMutableArray<MTLArgumentDescriptor*>* args,
															 MTLDataType dataType,
															 MTLArgumentAccess access,
															 mvk::SPIRVToMSLConversionConfiguration& shaderConfig,
															 uint32_t descSetIdx,
															 uint32_t argIdxOffset) {

	NSUInteger mtlArgDescAryLen = ((_info.descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT &&
									MVKInlineUniformBlockDescriptor::shouldEmbedInlineBlocksInMetalAgumentBuffer())
								   ? _info.descriptorCount : getDescriptorCount());

	auto* argDesc = [MTLArgumentDescriptor argumentDescriptor];
	argDesc.dataType = dataType;
	argDesc.access = access;
	argDesc.index = _argumentBufferIndex + argIdxOffset;
	argDesc.arrayLength = mtlArgDescAryLen;
	argDesc.textureType = shaderConfig.getMTLTextureType(descSetIdx, getBinding());

	[args addObject: argDesc];
}

// If depth compare is required, but unavailable on the device, the sampler can only be used as an immutable sampler
bool MVKDescriptorSetLayoutBinding::validate(MVKSampler* mvkSampler) {
	if (mvkSampler->getRequiresConstExprSampler()) {
		mvkSampler->reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkCmdPushDescriptorSet/vkCmdPushDescriptorSetWithTemplate(): Tried to push an immutable sampler.");
		return false;
	}
	return true;
}

void MVKDescriptorSetLayoutBinding::populateShaderConverterContext(mvk::SPIRVToMSLConversionConfiguration& shaderConfig,
                                                                   MVKShaderResourceBinding& dslMTLRezIdxOffsets,
                                                                   uint32_t dslIndex) {

	MVKSampler* mvkSamp = !_immutableSamplers.empty() ? _immutableSamplers.front() : nullptr;

    // Establish the resource indices to use, by combining the offsets of the DSL and this DSL binding.
    MVKShaderResourceBinding mtlIdxs = _mtlResourceIndexOffsets + dslMTLRezIdxOffsets;

    static const spv::ExecutionModel models[] = {
        spv::ExecutionModelVertex,
        spv::ExecutionModelTessellationControl,
        spv::ExecutionModelTessellationEvaluation,
        spv::ExecutionModelFragment,
        spv::ExecutionModelGLCompute
    };
    for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
        if (_applyToStage[i]) {
            mvkPopulateShaderConverterContext(shaderConfig,
                                              mtlIdxs.stages[i],
                                              models[i],
                                              dslIndex,
                                              _info.binding,
											  getDescriptorCount(),
											  mvkSamp);

			// If Metal argument buffers are in use, identify any inline uniform block bindings.
			if (_info.descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT &&
				isUsingMetalArgumentBuffer() &&
				MVKInlineUniformBlockDescriptor::shouldEmbedInlineBlocksInMetalAgumentBuffer()) {

				mvk::DescriptorBinding db;
				db.descriptorSet = dslIndex;
				db.binding = _info.binding;
				shaderConfig.inlineUniformBlocks.push_back(db);
			}
		}
    }
}

MVKDescriptorSetLayoutBinding::MVKDescriptorSetLayoutBinding(MVKDevice* device,
															 MVKDescriptorSetLayout* layout,
															 const VkDescriptorSetLayoutBinding* pBinding,
															 VkDescriptorBindingFlagsEXT bindingFlags,
															 uint32_t descriptorIndex) :
	MVKBaseDeviceObject(device),
	_layout(layout),
	_info(*pBinding),
	_flags(bindingFlags),
	_descriptorIndex(descriptorIndex) {

	_info.pImmutableSamplers = nullptr;     // Remove dangling pointer

	// Determine which shader stages this binding is used by, update the corresponding
	// shader stage selection in the layout, and establish the resource index offsets.
	for (uint32_t stage = kMVKShaderStageVertex; stage < kMVKShaderStageCount; stage++) {
        _applyToStage[stage] = mvkAreAllFlagsEnabled(pBinding->stageFlags, mvkVkShaderStageFlagBitsFromMVKShaderStage(MVKShaderStage(stage)));
		layout->_applyToStage[stage] |= _applyToStage[stage];
		initMetalResourceIndexOffsets(&_mtlResourceIndexOffsets.stages[stage],
									  &layout->_mtlResourceCounts.stages[stage],
									  pBinding, stage);
    }

    // If immutable samplers are defined, copy them in
    if ( pBinding->pImmutableSamplers &&
        (pBinding->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
         pBinding->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) ) {

		_immutableSamplers.reserve(pBinding->descriptorCount);
		for (uint32_t i = 0; i < pBinding->descriptorCount; i++) {
			_immutableSamplers.push_back((MVKSampler*)pBinding->pImmutableSamplers[i]);
			_immutableSamplers.back()->retain();
		}
	}
}

MVKDescriptorSetLayoutBinding::MVKDescriptorSetLayoutBinding(const MVKDescriptorSetLayoutBinding& binding) :
	MVKBaseDeviceObject(binding._device),
	_layout(binding._layout),
	_info(binding._info),
	_flags(binding._flags),
	_descriptorIndex(binding._descriptorIndex),
	_immutableSamplers(binding._immutableSamplers),
	_mtlResourceIndexOffsets(binding._mtlResourceIndexOffsets) {

	for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
        _applyToStage[i] = binding._applyToStage[i];
    }
	for (MVKSampler* sampler : _immutableSamplers) {
		sampler->retain();
	}
}

MVKDescriptorSetLayoutBinding::~MVKDescriptorSetLayoutBinding() {
	for (MVKSampler* sampler : _immutableSamplers) {
		sampler->release();
	}
}

// Sets the appropriate Metal resource indexes within this binding from the
// specified descriptor set binding counts, and updates those counts accordingly.
void MVKDescriptorSetLayoutBinding::initMetalResourceIndexOffsets(MVKShaderStageResourceBinding* pBindingIndexes,
																  MVKShaderStageResourceBinding* pDescSetCounts,
																  const VkDescriptorSetLayoutBinding* pBinding,
																  uint32_t stage) {

	// Sets an index offset and updates both that index and the general resource index.
	// Can be used multiply for combined multi-resource descriptor types.
	// When using Metal argument buffers, we accumulate the resource indexes cummulatively, across all resource types.
#	define setResourceIndexOffset(rezIdx) \
	do { \
		if (_applyToStage[stage] || isUsingMetalArgumentBuffer()) { \
			pBindingIndexes->rezIdx = isUsingMetalArgumentBuffer() ?  pDescSetCounts->resourceIndex : pDescSetCounts->rezIdx; \
			pDescSetCounts->rezIdx += descCnt; \
			pBindingIndexes->resourceIndex = pDescSetCounts->resourceIndex; \
			pDescSetCounts->resourceIndex += descCnt; \
		} \
	} while(false)

	// Only perform validation if this binding is used by the stage
#	define breakIfUnused() if ( !_applyToStage[stage] ) break

	uint32_t descCnt = pBinding->descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT ? 1 : pBinding->descriptorCount;
    switch (pBinding->descriptorType) {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
			setResourceIndexOffset(samplerIndex);
			breakIfUnused();

			if (pBinding->descriptorCount > 1 && !_device->_pMetalFeatures->arrayOfSamplers) {
				_layout->setConfigurationResult(reportError(VK_ERROR_FEATURE_NOT_PRESENT, "Device %s does not support arrays of samplers.", _device->getName()));
			}
            break;

        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
			setResourceIndexOffset(textureIndex);
			setResourceIndexOffset(samplerIndex);
			breakIfUnused();

			if (pBinding->descriptorCount > 1) {
				if ( !_device->_pMetalFeatures->arrayOfTextures ) {
					_layout->setConfigurationResult(reportError(VK_ERROR_FEATURE_NOT_PRESENT, "Device %s does not support arrays of textures.", _device->getName()));
				}
				if ( !_device->_pMetalFeatures->arrayOfSamplers ) {
					_layout->setConfigurationResult(reportError(VK_ERROR_FEATURE_NOT_PRESENT, "Device %s does not support arrays of samplers.", _device->getName()));
				}
			}

            if ( pBinding->pImmutableSamplers ) {
                for (uint32_t i = 0; i < pBinding->descriptorCount; i++) {
                    uint8_t planeCount = ((MVKSampler*)pBinding->pImmutableSamplers[i])->getPlaneCount();
                    if (planeCount > 1) {
                        pDescSetCounts->textureIndex += planeCount - 1;
                    }
                }
            }
            break;

        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
			setResourceIndexOffset(textureIndex);
			breakIfUnused();

			if (pBinding->descriptorCount > 1 && !_device->_pMetalFeatures->arrayOfTextures) {
				_layout->setConfigurationResult(reportError(VK_ERROR_FEATURE_NOT_PRESENT, "Device %s does not support arrays of textures.", _device->getName()));
			}
            break;

		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
		case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
			setResourceIndexOffset(textureIndex);
			setResourceIndexOffset(bufferIndex);
			breakIfUnused();

			if (pBinding->descriptorCount > 1 && !_device->_pMetalFeatures->arrayOfTextures) {
				_layout->setConfigurationResult(reportError(VK_ERROR_FEATURE_NOT_PRESENT, "Device %s does not support arrays of textures.", _device->getName()));
			}
			break;

        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
		case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
			setResourceIndexOffset(bufferIndex);
            break;

        default:
            break;
    }
}


#pragma mark -
#pragma mark MVKDescriptor

MTLResourceUsage MVKDescriptor::getMTLResourceUsage() {
	MTLResourceUsage mtlUsage = MTLResourceUsageRead;
	switch (getDescriptorType()) {
		case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
		case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
			mtlUsage |= MTLResourceUsageSample;
			break;

		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
		case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
			mtlUsage |= MTLResourceUsageWrite;
			break;

		default:
			break;
	}
	return mtlUsage;
}


#pragma mark -
#pragma mark MVKBufferDescriptor

// A null cmdEncoder can be passed to perform a validation pass
void MVKBufferDescriptor::bind(MVKCommandEncoder* cmdEncoder,
							   uint32_t descSetIndex,
							   uint32_t elementIndex,
							   MVKDescriptorSetLayoutBinding* mvkDSLBind,
							   bool stages[],
							   MVKShaderResourceBinding& mtlIndexes,
							   MVKArrayRef<uint32_t> dynamicOffsets,
							   uint32_t& dynamicOffsetIndex) {
	NSUInteger bufferDynamicOffset = 0;
	VkDescriptorType descType = getDescriptorType();
	if (descType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
		descType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
		if (dynamicOffsets.size > dynamicOffsetIndex) {
			bufferDynamicOffset = dynamicOffsets[dynamicOffsetIndex++];
		}
	}

	MVKMTLBufferBinding bb;
	bb.descriptorSetIndex = descSetIndex;
	bb.descriptorIndex = mvkDSLBind->getDescriptorIndex(elementIndex);
	bb.mtlUsage = getMTLResourceUsage();
	bb.mtlStages = mvkMTLRenderStagesFromMVKShaderStages(stages);

	if (_mvkBuffer) {
		bb.mtlBuffer = _mvkBuffer->getMTLBuffer();
		bb.offset = _mvkBuffer->getMTLBufferOffset() + _buffOffset + bufferDynamicOffset;
		if (_buffRange == VK_WHOLE_SIZE)
			bb.size = (uint32_t)(_mvkBuffer->getByteCount() - bb.offset);
		else
			bb.size = (uint32_t)_buffRange;
	}

	if (mvkDSLBind->isUsingMetalArgumentBuffer()) {
		bb.useArgumentBuffer = true;
		bb.index = mvkDSLBind->getMTLArgumentBufferIndex(elementIndex);
	}

	for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
		if (stages[i]) {
			if ( !bb.useArgumentBuffer ) {
				bb.index = mtlIndexes.stages[i].bufferIndex + elementIndex;
			}
			if (cmdEncoder) { cmdEncoder->bindBuffer(bb, MVKShaderStage(i)); }
		}
	}
}

void MVKBufferDescriptor::write(MVKDescriptorSetLayoutBinding* mvkDSLBind,
								MVKDescriptorSet* descSet,
								uint32_t srcIndex,
								uint32_t dstIndex,
								size_t stride,
								const void* pData) {
	auto* oldBuff = _mvkBuffer;

	const auto* pBuffInfo = &get<VkDescriptorBufferInfo>(pData, stride, srcIndex);
	_mvkBuffer = (MVKBuffer*)pBuffInfo->buffer;
	_buffOffset = pBuffInfo->offset;
	_buffRange = pBuffInfo->range;

	if (_mvkBuffer) { _mvkBuffer->retain(); }
	if (oldBuff) { oldBuff->release(); }
}

void MVKBufferDescriptor::read(uint32_t dstIndex,
							   VkDescriptorImageInfo* pImageInfo,
							   VkDescriptorBufferInfo* pBufferInfo,
							   VkBufferView* pTexelBufferView,
							   VkWriteDescriptorSetInlineUniformBlockEXT* pInlineUniformBlock) {
	auto& buffInfo = pBufferInfo[dstIndex];
	buffInfo.buffer = (VkBuffer)_mvkBuffer;
	buffInfo.offset = _buffOffset;
	buffInfo.range = _buffRange;
}

void MVKBufferDescriptor::reset() {
	if (_mvkBuffer) { _mvkBuffer->release(); }
	_mvkBuffer = nullptr;
	_buffOffset = 0;
	_buffRange = 0;
	MVKDescriptor::reset();
}


#pragma mark -
#pragma mark MVKInlineUniformBlockDescriptor

// A null cmdEncoder can be passed to perform a validation pass
void MVKInlineUniformBlockDescriptor::bind(MVKCommandEncoder* cmdEncoder,
										   uint32_t descSetIndex,
										   uint32_t elementIndex,
										   MVKDescriptorSetLayoutBinding* mvkDSLBind,
										   bool stages[],
										   MVKShaderResourceBinding& mtlIndexes,
										   MVKArrayRef<uint32_t> dynamicOffsets,
										   uint32_t& dynamicOffsetIndex) {

	MVKMTLBufferBinding bb;
	bb.descriptorSetIndex = descSetIndex;
	bb.descriptorIndex = mvkDSLBind->getDescriptorIndex(elementIndex);
	bb.mtlUsage = getMTLResourceUsage();
	bb.mtlStages = mvkMTLRenderStagesFromMVKShaderStages(stages);

	if (_isUsingIntermediaryMTLBuffer) {
		bb.isInline = false;
		auto* mtlBuffAlloc = (MVKMTLBufferAllocation*)_buffer;
		if (mtlBuffAlloc) {
			bb.mtlBuffer = mtlBuffAlloc->_mtlBuffer;
			bb.offset = mtlBuffAlloc->_offset;
		}
	} else {
		bb.isInline = true;
		bb.mtlBytes = getData();
	}
	bb.size = _length;

	if (mvkDSLBind->isUsingMetalArgumentBuffer()) {
		bb.useArgumentBuffer = true;
		bb.index = mvkDSLBind->getMTLArgumentBufferIndex();
	}

	for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
		if (stages[i]) {
			if ( !bb.useArgumentBuffer ) {
				bb.index = mtlIndexes.stages[i].bufferIndex;
			}
			if (cmdEncoder) { cmdEncoder->bindBuffer(bb, MVKShaderStage(i)); }
		}
	}
}

void MVKInlineUniformBlockDescriptor::write(MVKDescriptorSetLayoutBinding* mvkDSLBind,
											MVKDescriptorSet* descSet,
											uint32_t dstOffset,
											uint32_t dstIndex,
											size_t stride,
											const void* pData) {
	// Ensure there is a destination to write to
	if ( !_buffer ) {
		_length = mvkDSLBind->_info.descriptorCount;
		_isUsingIntermediaryMTLBuffer = mvkDSLBind->supportsMetalArgumentBuffers() && !shouldEmbedInlineBlocksInMetalAgumentBuffer();
		if (_isUsingIntermediaryMTLBuffer) {
			// Acquire an intermediary buffer and write it to the Metal argument buffer
			_buffer = (void*)descSet->acquireMTLBufferRegion(_length);
		} else {
			_buffer = malloc(_length);
		}
	}

	const auto& pInlineUniformBlock = *(VkWriteDescriptorSetInlineUniformBlockEXT*)pData;
	uint8_t* data = getData();
	if (data && pInlineUniformBlock.pData && dstOffset < _length) {
		uint32_t dataLen = std::min(pInlineUniformBlock.dataSize, _length - dstOffset);
		memcpy(data + dstOffset, pInlineUniformBlock.pData, dataLen);
	}
}

void MVKInlineUniformBlockDescriptor::read(uint32_t srcOffset,
                                           VkDescriptorImageInfo* pImageInfo,
                                           VkDescriptorBufferInfo* pBufferInfo,
                                           VkBufferView* pTexelBufferView,
                                           VkWriteDescriptorSetInlineUniformBlockEXT* pInlineUniformBlock) {
	uint8_t* data = getData();
	if (data && pInlineUniformBlock->pData && srcOffset < _length) {
		uint32_t dataLen = std::min(pInlineUniformBlock->dataSize, _length - srcOffset);
		memcpy((void*)pInlineUniformBlock->pData, data + srcOffset, dataLen);
	}
}

void MVKInlineUniformBlockDescriptor::reset() {
	if (_isUsingIntermediaryMTLBuffer) {
		if (_buffer) { ((MVKMTLBufferAllocation*)_buffer)->returnToPool(); }
	} else {
		free(_buffer);
	}
	_buffer = nullptr;
    _length = 0;
	_isUsingIntermediaryMTLBuffer = false;
	MVKDescriptor::reset();
}

uint8_t* MVKInlineUniformBlockDescriptor::getData() {
	return (uint8_t*)((_isUsingIntermediaryMTLBuffer && _buffer) ? ((MVKMTLBufferAllocation*)_buffer)->getContents() : _buffer);
}

// We do this once lazily instead of in a library constructor function to
// ensure the NSProcessInfo environment is available when called upon.
bool MVKInlineUniformBlockDescriptor::shouldEmbedInlineBlocksInMetalAgumentBuffer() {
	static bool _shouldEmbedInlineBlocksInMetalAgumentBuffer = MVK_CONFIG_EMBED_INLINE_BLOCKS_IN_METAL_ARGUMENT_BUFFER;
	static bool _shouldEmbedInlineBlocksInMetalAgumentBufferInitialized = false;
	if ( !_shouldEmbedInlineBlocksInMetalAgumentBufferInitialized ) {
		_shouldEmbedInlineBlocksInMetalAgumentBufferInitialized = true;
		MVK_SET_FROM_ENV_OR_BUILD_BOOL(_shouldEmbedInlineBlocksInMetalAgumentBuffer, MVK_CONFIG_EMBED_INLINE_BLOCKS_IN_METAL_ARGUMENT_BUFFER);
	}
	return _shouldEmbedInlineBlocksInMetalAgumentBuffer;
}


#pragma mark -
#pragma mark MVKImageDescriptor

// A null cmdEncoder can be passed to perform a validation pass
void MVKImageDescriptor::bind(MVKCommandEncoder* cmdEncoder,
							  uint32_t descSetIndex,
							  uint32_t elementIndex,
							  MVKDescriptorSetLayoutBinding* mvkDSLBind,
							  bool stages[],
							  MVKShaderResourceBinding& mtlIndexes,
							  MVKArrayRef<uint32_t> dynamicOffsets,
							  uint32_t& dynamicOffsetIndex) {
	MVKMTLTextureBinding tb;
	tb.descriptorSetIndex = descSetIndex;
	tb.descriptorIndex = mvkDSLBind->getDescriptorIndex(elementIndex);
	tb.mtlUsage = getMTLResourceUsage();
	tb.mtlStages = mvkMTLRenderStagesFromMVKShaderStages(stages);

	MVKMTLBufferBinding bb;
	bb.descriptorSetIndex = tb.descriptorSetIndex;
	bb.descriptorIndex = tb.descriptorIndex;
	bb.mtlUsage = tb.mtlUsage;
	bb.mtlStages = tb.mtlStages;

	VkDescriptorType descType = getDescriptorType();
	tb.swizzle = ((descType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
				   descType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) &&
				  tb.mtlTexture) ? _mvkImageView->getPackedSwizzle() : 0;

	uint32_t planeCount = _mvkImageView ? _mvkImageView->getPlaneCount() : 1;
	uint32_t atomBuffArgIdxOffset = mvkDSLBind->getDescriptorCount() * planeCount;

	for (uint8_t planeIndex = 0; planeIndex < planeCount; planeIndex++) {
		tb.mtlTexture = _mvkImageView ? _mvkImageView->getMTLTexture(planeIndex) : nil;

		if (descType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
			id<MTLTexture> mtlTex = tb.mtlTexture;
			if (mtlTex.parentTexture) { mtlTex = mtlTex.parentTexture; }
			bb.mtlBuffer = mtlTex.buffer;
			bb.offset = mtlTex.bufferOffset;
			bb.size = (uint32_t)(mtlTex.height * mtlTex.bufferBytesPerRow);
		}

		uint32_t planeDescIdx = (elementIndex * planeCount) + planeIndex;

		if (mvkDSLBind->isUsingMetalArgumentBuffer()) {
			tb.useArgumentBuffer = true;
			tb.index = mvkDSLBind->getMTLArgumentBufferIndex(planeDescIdx);

			bb.useArgumentBuffer = true;
			bb.index = tb.index + atomBuffArgIdxOffset;
		}

		for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
			if (stages[i]) {
				if ( !tb.useArgumentBuffer ) {
					tb.index = mtlIndexes.stages[i].textureIndex + planeDescIdx;
				}
				if (cmdEncoder) { cmdEncoder->bindTexture(tb, MVKShaderStage(i)); }

				if (descType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
					if ( !bb.useArgumentBuffer ) {
						bb.index = mtlIndexes.stages[i].bufferIndex + planeDescIdx;
					}
					if (cmdEncoder) { cmdEncoder->bindBuffer(bb, MVKShaderStage(i)); }
				}
			}
		}
	}
}

void MVKImageDescriptor::write(MVKDescriptorSetLayoutBinding* mvkDSLBind,
							   MVKDescriptorSet* descSet,
							   uint32_t srcIndex,
							   uint32_t dstIndex,
							   size_t stride,
							   const void* pData) {
	auto* oldImgView = _mvkImageView;

	const auto* pImgInfo = &get<VkDescriptorImageInfo>(pData, stride, srcIndex);
	_mvkImageView = (MVKImageView*)pImgInfo->imageView;
	_imageLayout = pImgInfo->imageLayout;

	if (_mvkImageView) { _mvkImageView->retain(); }
	if (oldImgView) { oldImgView->release(); }
}

void MVKImageDescriptor::read(uint32_t dstIndex,
							  VkDescriptorImageInfo* pImageInfo,
							  VkDescriptorBufferInfo* pBufferInfo,
							  VkBufferView* pTexelBufferView,
							  VkWriteDescriptorSetInlineUniformBlockEXT* pInlineUniformBlock) {
	auto& imgInfo = pImageInfo[dstIndex];
	imgInfo.imageView = (VkImageView)_mvkImageView;
	imgInfo.imageLayout = _imageLayout;
}

void MVKImageDescriptor::reset() {
	if (_mvkImageView) { _mvkImageView->release(); }
	_mvkImageView = nullptr;
	_imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	MVKDescriptor::reset();
}


#pragma mark -
#pragma mark MVKSamplerDescriptorMixin

// A null cmdEncoder can be passed to perform a validation pass
// Metal validation requires each sampler in an array of samplers to be populated,
// even if not used, so populate a default if one hasn't been set.
void MVKSamplerDescriptorMixin::bind(MVKCommandEncoder* cmdEncoder,
									 uint32_t descSetIndex,
									 uint32_t elementIndex,
									 MVKDescriptorSetLayoutBinding* mvkDSLBind,
									 bool stages[],
									 MVKShaderResourceBinding& mtlIndexes,
									 MVKArrayRef<uint32_t> dynamicOffsets,
									 uint32_t& dynamicOffsetIndex) {
	MVKMTLSamplerStateBinding sb;
	sb.descriptorSetIndex = descSetIndex;
	sb.descriptorIndex = mvkDSLBind->getDescriptorIndex(elementIndex);

	sb.mtlSamplerState = (_mvkSampler
						  ? _mvkSampler->getMTLSamplerState()
						  : cmdEncoder->getDevice()->getDefaultMTLSamplerState());

	if (mvkDSLBind->isUsingMetalArgumentBuffer()) {
		sb.useArgumentBuffer = true;
		sb.index = mvkDSLBind->getMTLArgumentBufferIndex(getSamplerArgBufferIndexOffset(mvkDSLBind) + elementIndex);
	}

	for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
		if (stages[i]) {
			if ( !sb.useArgumentBuffer ) {
				sb.index = mtlIndexes.stages[i].samplerIndex + elementIndex;
			}
			if (cmdEncoder) { cmdEncoder->bindSamplerState(sb, MVKShaderStage(i)); }
		}
	}
}

void MVKSamplerDescriptorMixin::write(MVKDescriptorSetLayoutBinding* mvkDSLBind,
									  MVKDescriptorSet* descSet,
									  uint32_t srcIndex,
									  uint32_t dstIndex,
									  size_t stride,
									  const void* pData) {
	if (_hasDynamicSampler) {
		auto* oldSamp = _mvkSampler;

		const auto* pImgInfo = &get<VkDescriptorImageInfo>(pData, stride, srcIndex);
		_mvkSampler = (MVKSampler*)pImgInfo->sampler;
		if (_mvkSampler && _mvkSampler->getRequiresConstExprSampler()) {
			_mvkSampler->reportError(VK_ERROR_FEATURE_NOT_PRESENT, "vkUpdateDescriptorSets(): Tried to push an immutable sampler.");
		}

		if (_mvkSampler) { _mvkSampler->retain(); }
		if (oldSamp) { oldSamp->release(); }
	}
}

void MVKSamplerDescriptorMixin::read(uint32_t dstIndex,
									 VkDescriptorImageInfo* pImageInfo,
									 VkDescriptorBufferInfo* pBufferInfo,
									 VkBufferView* pTexelBufferView,
									 VkWriteDescriptorSetInlineUniformBlockEXT* pInlineUniformBlock) {
	auto& imgInfo = pImageInfo[dstIndex];
	imgInfo.sampler = _hasDynamicSampler ? (VkSampler)_mvkSampler : nullptr;
}

// If the descriptor set layout binding contains immutable samplers, use them
// Otherwise the sampler will be populated dynamically at a later time.
void MVKSamplerDescriptorMixin::setLayout(MVKDescriptorSetLayoutBinding* dslBinding, uint32_t index) {
	auto* oldSamp = _mvkSampler;

	_mvkSampler = dslBinding->getImmutableSampler(index);
	_hasDynamicSampler = !_mvkSampler;

	if (_mvkSampler) { _mvkSampler->retain(); }
	if (oldSamp) { oldSamp->release(); }
}

void MVKSamplerDescriptorMixin::reset() {
	if (_mvkSampler) { _mvkSampler->release(); }
	_mvkSampler = nullptr;
	_hasDynamicSampler = true;
}


#pragma mark -
#pragma mark MVKSamplerDescriptor

// A null cmdEncoder can be passed to perform a validation pass
void MVKSamplerDescriptor::bind(MVKCommandEncoder* cmdEncoder,
								uint32_t descSetIndex,
								uint32_t elementIndex,
								MVKDescriptorSetLayoutBinding* mvkDSLBind,
								bool stages[],
								MVKShaderResourceBinding& mtlIndexes,
								MVKArrayRef<uint32_t> dynamicOffsets,
								uint32_t& dynamicOffsetIndex) {
	MVKSamplerDescriptorMixin::bind(cmdEncoder, descSetIndex, elementIndex, mvkDSLBind,
									stages, mtlIndexes, dynamicOffsets, dynamicOffsetIndex);
}

void MVKSamplerDescriptor::write(MVKDescriptorSetLayoutBinding* mvkDSLBind,
								 MVKDescriptorSet* descSet,
								 uint32_t srcIndex,
								 uint32_t dstIndex,
								 size_t stride,
								 const void* pData) {
	MVKSamplerDescriptorMixin::write(mvkDSLBind, descSet, srcIndex, dstIndex, stride, pData);
}

void MVKSamplerDescriptor::read(uint32_t dstIndex,
								VkDescriptorImageInfo* pImageInfo,
								VkDescriptorBufferInfo* pBufferInfo,
								VkBufferView* pTexelBufferView,
								VkWriteDescriptorSetInlineUniformBlockEXT* pInlineUniformBlock) {
	MVKSamplerDescriptorMixin::read(dstIndex, pImageInfo, pBufferInfo, pTexelBufferView, pInlineUniformBlock);
}

void MVKSamplerDescriptor::setLayout(MVKDescriptorSetLayoutBinding* dslBinding, uint32_t index) {
	MVKDescriptor::setLayout(dslBinding, index);
	MVKSamplerDescriptorMixin::setLayout(dslBinding, index);
}

void MVKSamplerDescriptor::reset() {
	MVKSamplerDescriptorMixin::reset();
	MVKDescriptor::reset();
}


#pragma mark -
#pragma mark MVKCombinedImageSamplerDescriptor

// A null cmdEncoder can be passed to perform a validation pass
void MVKCombinedImageSamplerDescriptor::bind(MVKCommandEncoder* cmdEncoder,
											 uint32_t descSetIndex,
											 uint32_t elementIndex,
											 MVKDescriptorSetLayoutBinding* mvkDSLBind,
											 bool stages[],
											 MVKShaderResourceBinding& mtlIndexes,
											 MVKArrayRef<uint32_t> dynamicOffsets,
											 uint32_t& dynamicOffsetIndex) {
	MVKImageDescriptor::bind(cmdEncoder, descSetIndex, elementIndex, mvkDSLBind,
							 stages, mtlIndexes, dynamicOffsets, dynamicOffsetIndex);
	MVKSamplerDescriptorMixin::bind(cmdEncoder, descSetIndex, elementIndex, mvkDSLBind,
									stages, mtlIndexes, dynamicOffsets, dynamicOffsetIndex);
}

void MVKCombinedImageSamplerDescriptor::write(MVKDescriptorSetLayoutBinding* mvkDSLBind,
											  MVKDescriptorSet* descSet,
											  uint32_t srcIndex,
											  uint32_t dstIndex,
											  size_t stride,
											  const void* pData) {
	MVKImageDescriptor::write(mvkDSLBind, descSet, srcIndex, dstIndex, stride, pData);
	MVKSamplerDescriptorMixin::write(mvkDSLBind, descSet, srcIndex, dstIndex, stride, pData);
}

void MVKCombinedImageSamplerDescriptor::read(uint32_t dstIndex,
											 VkDescriptorImageInfo* pImageInfo,
											 VkDescriptorBufferInfo* pBufferInfo,
											 VkBufferView* pTexelBufferView,
											 VkWriteDescriptorSetInlineUniformBlockEXT* pInlineUniformBlock) {
	MVKImageDescriptor::read(dstIndex, pImageInfo, pBufferInfo, pTexelBufferView, pInlineUniformBlock);
	MVKSamplerDescriptorMixin::read(dstIndex, pImageInfo, pBufferInfo, pTexelBufferView, pInlineUniformBlock);
}

void MVKCombinedImageSamplerDescriptor::setLayout(MVKDescriptorSetLayoutBinding* dslBinding, uint32_t index) {
	MVKImageDescriptor::setLayout(dslBinding, index);
	MVKSamplerDescriptorMixin::setLayout(dslBinding, index);
}

uint32_t MVKCombinedImageSamplerDescriptor::getSamplerArgBufferIndexOffset(MVKDescriptorSetLayoutBinding* dslBinding) {
	uint32_t planeCount = _mvkImageView ? _mvkImageView->getPlaneCount() : 1;
	return dslBinding->getDescriptorCount() * planeCount;
}

void MVKCombinedImageSamplerDescriptor::reset() {
	MVKSamplerDescriptorMixin::reset();
	MVKImageDescriptor::reset();
}


#pragma mark -
#pragma mark MVKTexelBufferDescriptor

// A null cmdEncoder can be passed to perform a validation pass
void MVKTexelBufferDescriptor::bind(MVKCommandEncoder* cmdEncoder,
									uint32_t descSetIndex,
									uint32_t elementIndex,
									MVKDescriptorSetLayoutBinding* mvkDSLBind,
									bool stages[],
									MVKShaderResourceBinding& mtlIndexes,
									MVKArrayRef<uint32_t> dynamicOffsets,
									uint32_t& dynamicOffsetIndex) {
	MVKMTLTextureBinding tb;
	tb.descriptorSetIndex = descSetIndex;
	tb.descriptorIndex = mvkDSLBind->getDescriptorIndex(elementIndex);
	tb.mtlUsage = getMTLResourceUsage();
	tb.mtlStages = mvkMTLRenderStagesFromMVKShaderStages(stages);

	MVKMTLBufferBinding bb;
	bb.descriptorSetIndex = tb.descriptorSetIndex;
	bb.descriptorIndex = tb.descriptorIndex;
	bb.mtlUsage = tb.mtlUsage;
	bb.mtlStages = tb.mtlStages;

	VkDescriptorType descType = getDescriptorType();
	if (_mvkBufferView) {
		tb.mtlTexture = _mvkBufferView->getMTLTexture();
		if (descType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
			id<MTLTexture> mtlTex = tb.mtlTexture;
			bb.mtlBuffer = mtlTex.buffer;
			bb.offset = mtlTex.bufferOffset;
			bb.size = (uint32_t)(mtlTex.height * mtlTex.bufferBytesPerRow);
		}
	}

	if (mvkDSLBind->isUsingMetalArgumentBuffer()) {
		tb.useArgumentBuffer = true;
		tb.index = mvkDSLBind->getMTLArgumentBufferIndex(elementIndex);

		bb.useArgumentBuffer = true;
		bb.index = tb.index + mvkDSLBind->getDescriptorCount();
	}

	for (uint32_t i = kMVKShaderStageVertex; i < kMVKShaderStageCount; i++) {
		if (stages[i]) {
			if ( !tb.useArgumentBuffer ) {
				tb.index = mtlIndexes.stages[i].textureIndex + elementIndex;
			}
			if (cmdEncoder) { cmdEncoder->bindTexture(tb, MVKShaderStage(i)); }

			if (descType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
				if ( !bb.useArgumentBuffer ) {
					bb.index = mtlIndexes.stages[i].bufferIndex + elementIndex;
				}
				if (cmdEncoder) { cmdEncoder->bindBuffer(bb, MVKShaderStage(i)); }
			}
		}
	}
}

void MVKTexelBufferDescriptor::write(MVKDescriptorSetLayoutBinding* mvkDSLBind,
									 MVKDescriptorSet* descSet,
									 uint32_t srcIndex,
									 uint32_t dstIndex,
									 size_t stride,
									 const void* pData) {
	auto* oldBuffView = _mvkBufferView;

	const auto* pBuffView = &get<VkBufferView>(pData, stride, srcIndex);
	_mvkBufferView = (MVKBufferView*)*pBuffView;

	if (_mvkBufferView) { _mvkBufferView->retain(); }
	if (oldBuffView) { oldBuffView->release(); }
}

void MVKTexelBufferDescriptor::read(uint32_t dstIndex,
									VkDescriptorImageInfo* pImageInfo,
									VkDescriptorBufferInfo* pBufferInfo,
									VkBufferView* pTexelBufferView,
									VkWriteDescriptorSetInlineUniformBlockEXT* pInlineUniformBlock) {
	pTexelBufferView[dstIndex] = (VkBufferView)_mvkBufferView;
}

void MVKTexelBufferDescriptor::reset() {
	if (_mvkBufferView) { _mvkBufferView->release(); }
	_mvkBufferView = nullptr;
	MVKDescriptor::reset();
}
