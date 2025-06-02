package net.lab1024.sa.admin.module.system.CasualtyInfo.dao;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import net.lab1024.sa.admin.module.system.CasualtyInfo.domain.entity.CasualtyEntity;
import net.lab1024.sa.admin.module.system.CasualtyInfo.domain.form.CasualtyQueryForm;
import net.lab1024.sa.admin.module.system.CasualtyInfo.domain.vo.CasualtyVO;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.vo.SupportDisasterVO;
import net.lab1024.sa.admin.module.system.employee.domain.vo.EmployeeVO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Component;

import java.util.Collection;
import java.util.List;

@Mapper
@Component
public interface CasualtyDao extends BaseMapper<CasualtyEntity> {
    /**
     * 查询伤亡列表.
     * @param page
     * @param queryForm
     * @return
     */
    List<CasualtyVO> queryCasualty(Page page, @Param("queryForm")CasualtyQueryForm queryForm);

    /**
     * 查询删除并的伤亡信息
     * @param deletedFlag
     * @return
     */
    List<CasualtyVO> selectCasualtyByDeleted(@Param("deletedFlag")Boolean deletedFlag);

    /**
     * 通过ID查询
     * @param ID
     * @return
     */
    CasualtyEntity getByID(@Param("ID")String ID);

    /**
     * 通过地址查询.
     * @param location
     * @return
     */
    CasualtyEntity getByLocation(@Param("location")String location);

    /**
     * 通过伤亡类型进行查询.
     * @param label
     * @return
     */
    CasualtyEntity getByLabel(@Param("label") String label);

    /**
     * 根据伤亡数量来查询.
     * @param number
     * @return
     */
    CasualtyEntity getByNumber(@Param("number") Integer number);

    /**
     * 查询一系列信息
     * @param IDs
     * @return
     */
    CasualtyEntity getByIDs(@Param("IDs")Collection<String> IDs);

    /**
     * 查询全部信息.
     * @return
     */
    List<CasualtyVO> listAll();

    /**
     * 查询伤害信息.
     * @param ID
     * @return
     */
    CasualtyVO getCasualtyById(@Param("ID") String ID);

    /**
     * 获取一批.
     * @param IDs
     * @return
     */
    List<CasualtyVO> getCasualtyByIds(@Param("IDs") Collection<String> IDs);


    /**
     * 获取所有.
     * @param leaveFlag
     * @return
     */
    List<String> getCasualtyId(@Param("leaveFlag") Boolean leaveFlag);


}
