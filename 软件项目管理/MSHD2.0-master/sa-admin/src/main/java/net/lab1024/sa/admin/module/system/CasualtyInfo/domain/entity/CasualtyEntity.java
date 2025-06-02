package net.lab1024.sa.admin.module.system.CasualtyInfo.domain.entity;


import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;


/**
 * 伤亡 实体表
 *
 * @Author 1024创新实验室-主任: 卓大
 * @Date 2021-12-09 22:57:49
 * @Wechat zhuoda1024
 * @Email lab1024@163.com
 * @Copyright 1024创新实验室 （ https://1024lab.net ）
 */
@Data
@TableName("t_casualty")
public class CasualtyEntity {

    /**
     * 统一编码
     */
    @TableId(type = IdType.INPUT)
    private String ID;

    /**
     * 发生地点
     */
    private String Location;

    /**
     * 发生时间
     */

    private String Time;

    /**
     * 伤亡类型
     */

    private String Label;

    /**
     * 伤亡及失踪数量
     */

    private Integer Number;

    /**
     * 数据来源
     */
    private String Origin;

    /**
     * 载体形式
     */
    private String Carrier;

    /**
     * 是否删除0否 1是
     */
    private Boolean deletedFlag;

    /**
     * 备注
     */
    private String remark;

    private LocalDateTime updateTime;

    private LocalDateTime createTime;

}
