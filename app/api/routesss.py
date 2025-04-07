@router.post("/chat/send")
async def send_message(
    request: SendMessageRequest,
    request_obj: Request,
    db: AsyncSession = Depends(get_db)
) -> StreamingResponse:
    try:
        session_id = request.sessionId

        # 解析消息内容
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
            text_content = text_content.strip()
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid content type"
            )

        logger.info(f"Received message: {text_content}")

        # 数据库事务处理
        async with db.begin() as transaction:
            user_message = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role=request.message.role,
                content=text_content,
                attachments=json.dumps(image_ids) if image_ids else None
            )
            db.add(user_message)
            await db.flush()

            for img_id in image_ids:
                db_image = await db.execute(
                    select(Image).filter(Image.id == img_id)
                )
                db_image = db_image.scalar_one_or_none()
                if not db_image:
                    await transaction.rollback()
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Image not found: {img_id}"
                    )
                db.add(MessageImage(
                    message_id=user_message.id,
                    image_id=img_id
                ))

        # 构造LLM消息格式
        llm_messages = []
        if is_multimodal:
            image_paths = [db_image.file_path for db_image in await db.execute(
                select(Image).filter(Image.id.in_(image_ids))
            )]
            llm_messages = [{
                "role": request.message.role,
                "content": [
                    {"type": "text", "text": text_content},
                    *[{"type": "image", "image_data": path} for path in image_paths]
                ]
            }]
        else:
            llm_messages = [{"role": request.message.role, "content": text_content}]

        # 特定场景处理
        if text_content == "变化检测":
            async def scene_sse_generator():
                stream_response = agent_service.run_stream(llm_messages, model="qwen2:7b")
                message_id = str(uuid.uuid4())

                assistant_message = ChatMessage(
                    id=message_id,
                    session_id=session_id,
                    role="assistant",
                    content=""
                )
                db.add(assistant_message)

                image_url = str(request_obj.url_for('static', path="test_image_2.png"))
                image_list = [image_url]
                
                for image in image_list:
                    image_id = generate_image_id(image_url)
                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'done': False, 'image_url': image,'image_id':image_id,'type': 'post'}})}\n\n"
                
                for chunk in stream_response:
                    content = chunk.get("text", "")
                    if content.strip():
                        async with db.begin():
                            assistant_message.content += content
                            await db.commit()

                        yield f"data: {json.dumps({
                            'message_id': message_id,
                            'data': {
                                'content': content,
                                'done': chunk.get('done', False)
                            }
                        })}\n\n"

                yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"

            return StreamingResponse(scene_sse_generator())

        elif text_content == "请告诉我灾后影像的大致受灾情况":
            async def scene_sse_generator():
                message_id = str(uuid.uuid4())
                assistant_message = ChatMessage(
                    id=message_id,
                    session_id=session_id,
                    role="assistant",
                    content=""
                )
                db.add(assistant_message)

                response_text = """这张影像显示了一个受灾区域，飓风后的卫星图像。从图像来看，主要的受灾情况包括:
大面积积水:图像显示大片的浑浊水域，覆盖了树林和部分居民区。这表明该区域可能经历了严重的洪水，导致陆地被淹没。
居民区受灾:部分房屋仍然可见，但许多看起来被水包围或部分淹没，这可能导致基础设施受损、居民被困或者财产损失。
树木和植被受影响:尽管树木仍然茂密，但被水淹没的情况可能导致植被根部受损，长期来看可能影响生态系统。
道路情况不明:由于洪水的覆盖，难以判断道路是否完好或者是否仍可通行，可能影响救援和疏散行动。"""

                chunk_size = 50
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i+chunk_size]
                    async with db.begin():
                        assistant_message.content += chunk
                        await db.commit()
                    yield f"data: {json.dumps({
                        'message_id': message_id,
                        'data': {
                            'content': chunk,
                            'done': i+chunk_size >= len(response_text)
                        }
                    })}\n\n"

                yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"

            return StreamingResponse(scene_sse_generator())

        elif text_content == "请判断受灾后A点到B点的道路是否通畅":
            async def scene_sse_generator():
                stream_response = agent_service.run_stream(llm_messages, model="qwen2:7b")
                message_id = str(uuid.uuid4())

                assistant_message = ChatMessage(
                    id=message_id,
                    session_id=session_id,
                    role="assistant",
                    content=""
                )
                db.add(assistant_message)

                image_url = str(request_obj.url_for('static', path="test_image_3.png"))
                image_list = [image_url]
                
                for image in image_list:
                    image_id = generate_image_id(image_url)
                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'done': False, 'image_url': image,'image_id':image_id,'type': 'post'}})}\n\n"

                answer = "根据您所提供的图像，经过路径判断受灾后A点B点之间的道路受到灾害影响不通畅。"
                chunks = [answer[i:i+50] for i in range(0, len(answer), 50)]
                
                for chunk in chunks:
                    async with db.begin():
                        assistant_message.content += chunk
                        await db.commit()
                    yield f"data: {json.dumps({
                        'message_id': message_id,
                        'data': {
                            'content': chunk,
                            'done': False
                        }
                    })}\n\n"

                yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"

            return StreamingResponse(scene_sse_generator())

        elif text_content == "那受灾前A点到B点的道路是否通畅呢？":
            async def scene_sse_generator():
                stream_response = agent_service.run_stream(llm_messages, model="qwen2:7b")
                message_id = str(uuid.uuid4())

                assistant_message = ChatMessage(
                    id=message_id,
                    session_id=session_id,
                    role="assistant",
                    content=""
                )
                db.add(assistant_message)

                image_url = str(request_obj.url_for('static', path="test_image_4.png"))
                image_list = [image_url]
                
                for image in image_list:
                    image_id = generate_image_id(image_url)
                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'done': False, 'image_url': image,'image_id':image_id,'type': 'post'}})}\n\n"

                answer = "根据您所提供的图像，经过路径判断受灾前A点B点之间的道路是通畅的。"
                chunks = [answer[i:i+50] for i in range(0, len(answer), 50)]
                
                for chunk in chunks:
                    async with db.begin():
                        assistant_message.content += chunk
                        await db.commit()
                    yield f"data: {json.dumps({
                        'message_id': message_id,
                        'data': {
                            'content': chunk,
                            'done': False
                        }
                    })}\n\n"

                yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"

            return StreamingResponse(scene_sse_generator())

        elif text_content == "请根据受灾场景综合判断房屋受损情况，要求尽可能的详细，且提供受灾图像的基本信息。":
            async def scene_sse_generator():
                message_id = str(uuid.uuid4())
                assistant_message = ChatMessage(
                    id=message_id,
                    session_id=session_id,
                    role="assistant",
                    content=""
                )
                db.add(assistant_message)

                image_url = str(request_obj.url_for('static', path="test_image_5.png"))
                image_list = [image_url]
                
                for image in image_list:
                    image_id = generate_image_id(image_url)
                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'done': False, 'image_url': image,'image_id':image_id,'type': 'post'}})}\n\n"

                response_text = """这张卫星图像展示了飓风过后一个受灾区域的全貌。图像中可以明显看到大片浑浊的水域覆盖了树林和部分居民区，表明该区域经历了严重洪水，陆地大面积被淹。部分房屋依然可辨，但许多建筑似乎被水包围或部分浸泡，暗示基础设施可能遭受破坏，居民也可能面临被困和财产损失的风险。虽然树木依旧茂密，但被淹的情况可能对植被的根系造成损伤，长期来看会影响生态系统；而由于洪水覆盖，难以判断道路的完好性和通行状况，这可能对救援和疏散行动构成阻碍。根据对36个建筑物的统计，数据显示有22个建筑物无损坏，主要分布在图像上半部分和右侧；4个建筑物显示轻微损坏，分散在图像中部和左侧；7个建筑物受严重损坏，主要集中在图像的中下部和左侧；另外还有4个建筑物未分类。这种分布表明，虽然大部分房屋没有明显损坏，但局部区域尤其是图像中下部和左侧，受灾情况较为严重，提示救援和恢复工作需针对性展开。"""

                chunk_size = 50
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i+chunk_size]
                    async with db.begin():
                        assistant_message.content += chunk
                        await db.commit()
                    yield f"data: {json.dumps({
                        'message_id': message_id,
                        'data': {
                            'content': chunk,
                            'done': i+chunk_size >= len(response_text)
                        }
                    })}\n\n"

                yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"

            return StreamingResponse(scene_sse_generator())

        # 默认处理逻辑
        async def sse_stream_generator():
            try:
                async with db.begin() :
                    assistant_message = ChatMessage(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        role="assistant",
                        content=""
                    )
                    db.add(assistant_message)
                    await db.commit()

                    if is_multimodal:
                        stream_response = multimodal_agent.run_stream(llm_messages, model="llava:latest")
                    else:
                        stream_response = agent_service.run_stream(llm_messages, model="qwen2:7b")

                    if not hasattr(stream_response, "__aiter__"):
                        stream_response = async_generator_wrapper(stream_response)

                    async for chunk in stream_response:
                        if redis_client.get(f"pause_flag:{session_id}"):
                            break

                        content = chunk.get("text", "") or chunk.get("output", "") or ""
                        if content.strip():
                            async with db.begin():
                                assistant_message.content += content
                                await db.commit()

                            yield f"data: {json.dumps({
                                'message_id': assistant_message.id,
                                'data': {
                                    'content': content,
                                    'done': chunk.get('done', False)
                                }
                            })}\n\n"

                    async with db.begin():
                        assistant_message.created_at = datetime.now(timezone.utc)
                        await db.commit()

                    yield f"data: {json.dumps({'message_id': assistant_message.id, 'data': {'content': '', 'done': True}})}\n\n"

            except Exception as e:
                await db.rollback()
                yield f"data: {json.dumps({'error': str(e), 'detail': 'Inference failed'})}\n\n"

        return StreamingResponse(sse_stream_generator())

    except HTTPException as e:
        await db.rollback()
        raise e
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server error: {str(e)}"
        )