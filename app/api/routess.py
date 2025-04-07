@router.post("/chat/send")
async def send_message(
    request: SendMessageRequest,
    request_obj: Request,
    db: AsyncSession = Depends(get_db)
) -> StreamingResponse:
    try:
        session_id = request.sessionId

        # 2. 解析消息内容
        text_content = ""
        image_ids = []
        is_multimodal = False

        if isinstance(request.message.content, str):
            text_content = request.message.content.strip()
        elif isinstance(request.message.content, list):
            for item in request.message.content:
                if item.type == "text":
                    text_content += item.text.strip() + " "
                elif item.type == "image":
                    if not item.image_id:
                        raise HTTPException(
                            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Image content missing image_id"
                        )
                    image_ids.append(item.image_id)
                    is_multimodal = True
            text_content = text_content.strip()  # 去除多余空格
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid content type. Expected str or list"
            )
        logger.info(f"收到消息: {text_content}")  # 关键调试日志

        # 3. 数据库事务处理，确保用户信息先存入数据库
        async with db.begin() as transaction:
            # 主消息记录
            chat_message = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role=request.message.role,
                content=text_content,
                attachments=json.dumps(image_ids) if image_ids else None
            )
            db.add(chat_message)
            await db.flush()  # 确保chat_message.id生成

            # 验证并关联图片（异步查询）
            image_paths = []
            for img_id in image_ids:
                result = await db.execute(
                    select(Image).filter(Image.id == img_id)
                )
                db_image = result.scalar_one_or_none()
                if not db_image:
                    await transaction.rollback()
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Image not found: {img_id}"
                    )
                image_paths.append(db_image.file_path)  # 收集图片路径

                # 消息-图片关联
                db.add(MessageImage(
                    message_id=chat_message.id,
                    image_id=img_id
                ))
            logging.info(f"成功插入消息: {chat_message.id}, 关联图片数量: {len(image_ids)}")

        # 4. 构造LLM消息格式
        llm_messages = []
        if is_multimodal:
            # 多模态消息格式（文本+图片路径）
            llm_messages = [{
                "role": request.message.role,
                "content": [
                    {"type": "text", "text": text_content},
                    *[{"type": "image", "image_data": path} for path in image_paths]
                ]
            }]
        else:
            # 纯文本消息
            llm_messages = [{"role": request.message.role, "content": text_content}]

        # 5. 流式响应生成
        async def sse_stream_generator() -> Generator[str, None, None]:
            try:
                async with response_lock:
                    # 创建AI消息记录
                    pause_flag_key = f"pause_flag:{session_id}"  # 定义暂停标志键
                    
                    assistant_message = ChatMessage(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        role="assistant",
                        content="",  # 后续流式填充内容
                        attachments=None,
                    )

                    async with db.begin() as sub_transaction:
                        db.add(assistant_message)
                        await db.commit()

                    # 调用大模型（假设为异步调用）
                    if is_multimodal:
                        stream_response = multimodal_agent.run_stream(llm_messages, model="llava:latest")
                    else:
                        stream_response = agent_service.run_stream(llm_messages, model="qwen2:7b")

                    # 如果是同步生成器，包装为异步生成器
                    if not hasattr(stream_response, "__aiter__"):
                        stream_response = async_generator_wrapper(stream_response)

                    async for chunk in stream_response:

                        if redis_client.get(pause_flag_key):
                            logger.info(f"会话 {session_id} 收到暂停指令，终止流式回复")
                            break

                        content = (
                            chunk.get("message", {}).get("content", "")  # Ollama 格式
                        )
                        if not content:
                            content = chunk.get("text", "") or chunk.get("output", "") or ""
                        if content.strip():
                            async with db.begin() as update_transaction:
                                assistant_message.content += content
                                await db.commit()

                            sse_chunk = {
                                "message_id": assistant_message.id,
                                "data": {
                                    "content": content,
                                    "done": chunk.get("done", False)
                                }
                            }
                            # 发送SSE消息
                            yield f"data: {json.dumps(sse_chunk)}\n\n"

                    async with db.begin() as final_transaction:
                        assistant_message.created_at = datetime.now(timezone.utc)
                        await db.commit()

                    logger.info("流式回复生成完成")
                    yield f"data: {json.dumps({'message_id': assistant_message.id, 'data': {'content': '', 'done': True}})}\n\n"

            except Exception as e:
                await db.rollback()
                error_chunk = {
                    "error": str(e),
                    "detail": "Inference failed"
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(
            sse_stream_generator(),
            media_type="text/event-stream"
        )

    except HTTPException as e:
        await db.rollback()
        raise e

    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server error: {str(e)}"
        )